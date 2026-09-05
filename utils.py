import copy
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

import pm4py
import torch
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.log.obj import Event, EventLog, Trace
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.petri_net.common import visualize as pn_common_visualize


CONTROL_LABELS = {"start", "end", "__start__", "__end__"}
ADDED_PREFIXES = ("rl_rel_", "rl_frag_", "rl_align_", "rl_missing_", "rl_relax_", "rl_safe_")

@dataclass(frozen=True)
class Anchor:
    source_label: str
    target_label: str
    source_name: str
    target_name: str
    place_name: str

@dataclass
class Candidate:
    kind: str
    source_label: str
    target_label: str
    support: int
    inserted: Tuple[str, ...] = tuple()
    anchor: Optional[Anchor] = None
    source_transition_name: Optional[str] = None
    target_transition_name: Optional[str] = None
    source_place_name: Optional[str] = None
    target_place_name: Optional[str] = None
    remove_transition_name: Optional[str] = None
    remove_place_name: Optional[str] = None
    remove_arc_names: Tuple[Tuple[str, str], ...] = tuple()
    probability: float = 0.0
    score: float = 0.0

    @property
    def key(self) -> Tuple[str, str, Tuple[str, ...], str, Optional[str], Optional[str]]:
        anchor_id = self.anchor.place_name if self.anchor else None
        place_id = None
        if self.source_place_name or self.target_place_name:
            place_id = f"{self.source_place_name or ''}->{self.target_place_name or ''}"
        return (self.kind, self.source_label, self.inserted, self.target_label, anchor_id, place_id)

    @property
    def action_label(self) -> str:
        if self.kind == "safe_duplicate":
            return f"add silent duplicate route for {self.source_label}"
        if self.kind == "hidden_relation":
            return f"{self.source_label} -> {self.target_label}"
        if self.kind == "alignment_fragment":
            fragment = " -> ".join(self.inserted)
            return f"{self.source_label} => [{fragment}] => {self.target_label}"
        if self.kind == "missing_activity":
            fragment = " -> ".join(self.inserted)
            return f"insert missing [{fragment}] between {self.source_label} and {self.target_label}"
        if self.kind == "relaxed_transition":
            return f"relax synchronization for {self.source_label} via {self.source_place_name}"
        if self.kind == "tau_prune":
            return f"remove silent transition {self.remove_transition_name}"
        if self.kind == "arc_prune":
            return f"remove arc(s) {self.remove_arc_names}"
        fragment = " -> ".join(self.inserted)
        return f"{self.source_label} -> [{fragment}] -> {self.target_label}"

    @property
    def visible_duplicate_count(self) -> int:
        if self.kind in {"hidden_relation", "missing_activity", "tau_prune", "arc_prune", "safe_duplicate"}:
            return 0
        return len(self.inserted)

    @property
    def estimated_complexity(self) -> int:
        if self.kind == "safe_duplicate":
            return 1
        if self.kind == "relaxed_transition":
            return 1
        if self.kind == "hidden_relation":
            return 3
        if self.kind in {"tau_prune", "arc_prune"}:
            return -2
        if self.kind in {"alignment_fragment", "missing_activity"}:
            return len(self.inserted) * 2 + 1
        return len(self.inserted) * 2 + 1

@dataclass
class Metrics:
    fitness: float
    precision: float
    simplicity: float
    f1: float
    composite: float

@dataclass
class RepairDecision:
    step: int
    action: str
    candidate: Optional[Candidate]
    reward: float
    before: Metrics
    after: Metrics
    probability: float
    note: str

@dataclass
class CandidateBatch:
    relation_candidates: List[Candidate]
    fragment_candidates: List[Candidate]
    alignment_candidates: List[Candidate]
    relaxation_candidates: List[Candidate]
    prune_candidates: List[Candidate]
    candidates: List[Candidate]

@dataclass
class MDPTransition:
    state: torch.Tensor
    candidate_features: torch.Tensor
    action: int
    reward: float
    done: bool
    old_log_prob: torch.Tensor
    value: torch.Tensor
    next_value: torch.Tensor

def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

def canonicalize_label(raw: str, known_labels: Set[str]) -> Optional[str]:
    if not raw:
        return None
    if raw in known_labels:
        return raw
    replacements = (raw.replace("_", "-"), raw.replace("_", " "), raw.replace("-", "_"), raw.strip())
    for cand in replacements:
        if cand in known_labels:
            return cand
    return raw

def format_metric(value: float) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.6f}"

def metric_delta(after: Metrics, before: Metrics, key: str) -> float:
    return getattr(after, key) - getattr(before, key)

def read_xes_variants_streaming(xes_path: Path, known_labels: Set[str], max_traces: int = 0) -> Tuple[Counter, int, int]:
    variants = Counter()
    current_trace = []
    in_event = False
    event_name = None
    trace_count = 0
    event_count = 0
    for event, elem in ET.iterparse(str(xes_path), events=("start", "end")):
        tag = strip_namespace(elem.tag)
        if event == "start":
            if tag == "trace":
                current_trace = []
            elif tag == "event":
                in_event = True
                event_name = None
            continue
        if tag == "string" and in_event and elem.attrib.get("key") == "concept:name":
            event_name = canonicalize_label(elem.attrib.get("value", ""), known_labels)
            elem.clear()
        elif tag == "event":
            if event_name:
                current_trace.append(event_name)
                event_count += 1
            in_event = False
            event_name = None
            elem.clear()
        elif tag == "trace":
            if current_trace:
                variants[tuple(current_trace)] += 1
                trace_count += 1
                if max_traces and trace_count >= max_traces:
                    elem.clear()
                    break
            elem.clear()
        elif tag in {"date", "int", "float", "boolean"}:
            elem.clear()
    return variants, trace_count, event_count

def build_event_log_from_variants(variants: Counter, max_traces: int = 0) -> EventLog:
    log = EventLog()
    trace_idx = 0
    for variant, freq in variants.most_common():
        for _ in range(freq):
            if max_traces and trace_idx >= max_traces:
                return log
            trace = Trace(attributes={"concept:name": f"case_{trace_idx}"})
            for label in variant:
                trace.append(Event({"concept:name": label}))
            log.append(trace)
            trace_idx += 1
    return log

def build_sample_log_from_variants(variants: Counter, max_traces: int, min_variant_coverage: int = 0) -> EventLog:
    if max_traces <= 0:
        return build_event_log_from_variants(variants, 0)
    log = EventLog()
    trace_idx = 0
    selected_variants = variants.most_common(min_variant_coverage) if min_variant_coverage else []
    selected_keys = {v for v, _ in selected_variants}
    for variant, _ in selected_variants:
        if trace_idx >= max_traces:
            break
        trace = Trace(attributes={"concept:name": f"sample_{trace_idx}"})
        for label in variant:
            trace.append(Event({"concept:name": label}))
        log.append(trace)
        trace_idx += 1
    for variant, freq in variants.most_common():
        if trace_idx >= max_traces:
            break
        if variant in selected_keys:
            repeats = max(0, min(freq - 1, max_traces - trace_idx))
        else:
            repeats = min(freq, max_traces - trace_idx)
        for _ in range(repeats):
            trace = Trace(attributes={"concept:name": f"sample_{trace_idx}"})
            for label in variant:
                trace.append(Event({"concept:name": label}))
            log.append(trace)
            trace_idx += 1
    return log

def load_original_metrics_cache(report_path: Path) -> Optional[Metrics]:
    if not report_path.exists():
        return None
    values: Dict[str, float] = {}
    pattern = re.compile(r"^\|\s*(fitness|precision|f1|simplicity|composite)\s*\|\s*([-+0-9.]+)")
    for line in report_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    required = {"fitness", "precision", "simplicity", "f1", "composite"}
    if not required.issubset(values):
        return None
    return Metrics(values["fitness"], values["precision"], values["simplicity"],
                   values["f1"], values["composite"])

def log_directly_follows(variants: Counter, model_labels: Set[str]) -> Counter:
    dfg = Counter()
    for variant, freq in variants.items():
        filtered = [l for l in variant if l in model_labels]
        for left, right in zip(filtered, filtered[1:]):
            if left != right:
                dfg[(left, right)] += freq
    return dfg

def alignment_move_label(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, tuple) and value:
        for part in reversed(value):
            if part and part != ">>":
                return str(part)
        return None
    text = str(value)
    if text == ">>" or not text:
        return None
    return text

def export_pnml(net, im, fm, path):
    try:
        pm4py.write_pnml(net, im, fm, str(path))
    except:
        from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
        pnml_exporter.apply(net, im, str(path), final_marking=fm)

def write_report(out_path, args, trace_count, event_count, variants_count,
                 rel_cands, frag_cands, align_cands, relax_cands, prune_cands, ranked, selected, decisions,
                 orig_metrics, repaired_metrics, pnml_path):
    lines = [f"# {args.data} Repair Report (Standard MDP PPO)", ""]
    lines.append(f"- Traces: {trace_count}, Events: {event_count}, Variants: {variants_count}")
    lines.append(f"- Selected repairs: {len(selected)}")
    lines.append(f"- Fitness floor satisfied: {repaired_metrics.fitness >= orig_metrics.fitness - args.metric_tolerance}")
    lines.append(
        f"- Candidate pool: relation={len(rel_cands)}, anchor={len(frag_cands)}, "
        f"alignment={sum(c.kind == 'alignment_fragment' for c in align_cands)}, "
        f"missing_activity={sum(c.kind == 'missing_activity' for c in align_cands)}, "
        f"relaxation={len(relax_cands)}, "
        f"prune={len(prune_cands)}, action={len(ranked)}"
    )
    lines.append(
        f"- Search: max_steps={args.max_steps}, max_repairs={args.max_repairs}, "
        f"precision_validation_trials={args.precision_validation_trials}, "
        f"full_validation_top_k={args.full_validation_top_k}"
    )
    lines.append(f"- PNML: {pnml_path.name}")
    lines.append("")
    lines.append("| metric | original | repaired | delta |")
    lines.append("|---|---:|---:|---:|")
    for k in ("fitness", "precision", "f1", "simplicity", "composite"):
        delta = metric_delta(repaired_metrics, orig_metrics, k)
        lines.append(f"| {k} | {format_metric(getattr(orig_metrics,k))} | {format_metric(getattr(repaired_metrics,k))} | {format_metric(delta)} |")
    lines.append("")
    lines.append("## Applied Repairs")
    if selected:
        lines.append("| # | kind | action | support |")
        lines.append("|---:|---|---|---:|")
        for idx, cand in enumerate(selected, start=1):
            lines.append(f"| {idx} | {cand.kind} | {cand.action_label} | {cand.support} |")
    else:
        lines.append("No valid repair was applied.")
    lines.append("")
    lines.append("## MDP Decisions")
    lines.append("| step | decision | action | reward | probability | fitness | precision | f1 | composite | note |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    for decision in decisions:
        action = decision.candidate.action_label if decision.candidate else "-"
        safe_note = decision.note.replace("|", "/")
        lines.append(
            f"| {decision.step} | {decision.action} | {action} | {decision.reward:.6f} | "
            f"{decision.probability:.6f} | {decision.after.fitness:.6f} | {decision.after.precision:.6f} | "
            f"{decision.after.f1:.6f} | {decision.after.composite:.6f} | {safe_note} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")