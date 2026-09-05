import copy
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import Marking, PetriNet

from utils import *
from petri_net_ops import *

def outcome_label_score(label: Optional[str]) -> float:
    if not label:
        return 0.0
    upper = label.upper()
    outcome_terms = (
        "CANCEL", "DECLIN", "DENIED", "REJECT", "RETURN",
        "FINAL", "WITHDRAW", "REFUSE", "ABORT", "END"
    )
    return 1.0 if any(term in upper for term in outcome_terms) else 0.0

def mine_hidden_relation_candidates(dfg: Counter, footprint: Set[Tuple[str, str]], model_labels: Set[str],
                                    min_support: int, min_relative_support: float, trace_count: int, limit: int) -> List[Candidate]:
    candidates = []
    for (source, target), support in dfg.most_common():
        if source not in model_labels or target not in model_labels:
            continue
        if source in CONTROL_LABELS or target in CONTROL_LABELS:
            continue
        if (source, target) in footprint:
            continue
        if support < min_support:
            continue
        if trace_count and support / trace_count < min_relative_support:
            continue
        candidates.append(Candidate(kind="hidden_relation", source_label=source, target_label=target, support=support))
        if len(candidates) >= limit:
            break
    return candidates

def expand_hidden_relation_places(net: PetriNet, candidates: Sequence[Candidate], limit: int) -> List[Candidate]:
    expanded: List[Candidate] = []
    for cand in candidates:
        sources = transitions_by_label(net, cand.source_label)
        targets = transitions_by_label(net, cand.target_label)
        source_places = []
        target_places = []
        for source in sources[:4]:
            source_places.extend(
                arc.target for arc in source.out_arcs
                if isinstance(arc.target, PetriNet.Place)
            )
        for target in targets[:4]:
            target_places.extend(
                arc.source for arc in target.in_arcs
                if isinstance(arc.source, PetriNet.Place)
            )
        source_places = sorted(set(source_places), key=lambda p: (len(p.out_arcs), p.name))[:4]
        target_places = sorted(set(target_places), key=lambda p: (len(p.in_arcs), p.name))[:6]
        for source_place in source_places:
            for target_place in target_places:
                if source_place is target_place:
                    continue
                expanded.append(copy.copy(cand))
                expanded[-1].source_place_name = source_place.name
                expanded[-1].target_place_name = target_place.name
                if len(expanded) >= limit:
                    return expanded
    return expanded

def mine_anchor_fragment_candidates(variants: Counter, anchors_by_pair: Dict, net: PetriNet, model_labels: Set[str],
                                    max_insert_len: int, min_support: int, limit: int) -> List[Candidate]:
    counts = Counter()
    anchor_pairs = set(anchors_by_pair)
    for variant, freq in variants.items():
        seq = [l for l in variant if l in model_labels]
        for left, source in enumerate(seq):
            for ins_len in range(1, max_insert_len + 1):
                right = left + ins_len + 1
                if right >= len(seq):
                    break
                target = seq[right]
                if (source, target) not in anchor_pairs:
                    continue
                inserted = tuple(seq[left+1:right])
                if any(l in CONTROL_LABELS for l in inserted):
                    continue
                if inserted[0] == target or inserted[-1] == source:
                    continue
                counts[(source, inserted, target)] += freq
    candidates = []
    for (source, inserted, target), support in counts.most_common():
        if support < min_support:
            continue
        anchors = sorted(anchors_by_pair[(source, target)], key=lambda a: (a.place_name, a.source_name, a.target_name))
        for anchor in anchors:
            candidates.append(Candidate(kind="anchor_fragment", source_label=source, target_label=target,
                                        inserted=inserted, support=support, anchor=anchor))
            if len(candidates) >= limit:
                return candidates
    return candidates

def mine_alignment_fragment_candidates(log: EventLog, net: PetriNet, initial_marking: Marking, final_marking: Marking,
                                       model_labels: Set[str], min_support: int, max_insert_len: int, limit: int) -> List[Candidate]:
    counts = Counter()
    try:
        print('mine alignment fragment candidates')
        aligned = alignments.apply_log(copy.deepcopy(log), net, initial_marking, final_marking,
                                       variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
    except Exception as e:
        print(f"Alignment mining skipped: {e}")
        return []
    for result in aligned:
        prev_sync = None
        pending = []
        for log_move, model_move in result.get("alignment", []):
            log_label = alignment_move_label(log_move)
            model_label = alignment_move_label(model_move)
            is_sync = (log_label is not None and model_label is not None and log_label == model_label and log_label in model_labels)
            is_log_move = log_label is not None and (model_label is None or model_label == ">>")
            if is_sync:
                if prev_sync and pending and len(pending) <= max_insert_len:
                    counts[(prev_sync, tuple(pending), log_label)] += 1
                prev_sync = log_label
                pending = []
            elif is_log_move and prev_sync:
                pending.append(log_label)
                if len(pending) > max_insert_len:
                    pending = []
                    prev_sync = None
            elif model_label is not None:
                continue
    candidates = []
    for (source, inserted, target), support in counts.most_common():
        if support < min_support:
            continue
        if source in CONTROL_LABELS or target in CONTROL_LABELS:
            continue
        if any(l in CONTROL_LABELS for l in inserted):
            continue
        kind = "missing_activity" if any(label not in model_labels for label in inserted) else "alignment_fragment"
        candidates.append(Candidate(kind=kind, source_label=source, target_label=target,
                                    inserted=inserted, support=support))
        if len(candidates) >= limit:
            break
    return candidates

def mine_relaxed_transition_candidates(
    variants: Counter,
    net: PetriNet,
    min_support: int,
    limit: int,
) -> List[Candidate]:
    activity_support = Counter()
    for variant, frequency in variants.items():
        for label in variant:
            activity_support[label] += frequency
    candidates: List[Candidate] = []
    transitions = sorted(
        [transition for transition in net.transitions if transition.label is not None],
        key=lambda transition: (-activity_support[transition.label], transition.name),
    )
    for transition in transitions:
        input_places = sorted(
            [arc.source for arc in transition.in_arcs if isinstance(arc.source, PetriNet.Place)],
            key=lambda place: place.name,
        )
        if len(input_places) <= 1:
            continue
        support = activity_support[transition.label]
        if support < min_support:
            continue
        for input_place in input_places:
            candidates.append(Candidate(
                kind="relaxed_transition",
                source_label=transition.label,
                target_label=transition.label,
                support=support,
                source_transition_name=transition.name,
                source_place_name=input_place.name,
            ))
            if len(candidates) >= limit:
                return candidates
    return candidates

def mine_prune_candidates(net: PetriNet, limit: int) -> List[Candidate]:
    tau_candidates = []
    for trans in sorted(net.transitions, key=lambda t: t.name):
        if trans.label is not None:
            continue
        if not trans.in_arcs or not trans.out_arcs:
            continue
        tau_candidates.append(Candidate(kind="tau_prune", source_label="__prune__", target_label="__prune__",
                                        support=1, remove_transition_name=trans.name))
    arc_candidates = []
    for place in sorted(net.places, key=lambda p: p.name):
        incoming = sorted(place.in_arcs, key=lambda a: getattr(a.source, "name", ""))
        outgoing = sorted(place.out_arcs, key=lambda a: getattr(a.target, "name", ""))
        if len(incoming) <= 1 or len(outgoing) <= 1:
            continue
        for arc in incoming + outgoing:
            arc_candidates.append(Candidate(
                kind="arc_prune", source_label="__prune__", target_label="__prune__",
                support=1, remove_arc_names=((arc.source.name, arc.target.name),)))
    tau_quota = (limit + 1) // 2
    arc_quota = limit - tau_quota
    selected = tau_candidates[:tau_quota] + arc_candidates[:arc_quota]
    remainder = tau_candidates[tau_quota:] + arc_candidates[arc_quota:]
    return (selected + remainder)[:limit]

def feature_tensor(candidates: List[Candidate], trace_count: int) -> torch.Tensor:
    if not candidates:
        return torch.empty((0, 14), dtype=torch.float32)
    max_support = max(math.log1p(c.support) for c in candidates)
    max_complexity = max(abs(c.estimated_complexity) for c in candidates)
    endpoint_counts = Counter()
    for c in candidates:
        endpoint_counts[c.source_label] += 1
        endpoint_counts[c.target_label] += 1
    max_endpoint = max(endpoint_counts.values()) if endpoint_counts else 1
    rows = []
    for c in candidates:
        log_sup = math.log1p(c.support)
        relative = c.support / max(trace_count, 1)
        complexity = c.estimated_complexity
        endpoint_overlap = (endpoint_counts[c.source_label] + endpoint_counts[c.target_label] - 2) / max(max_endpoint, 1)
        rows.append([
            log_sup / max(max_support, 1.0),
            min(relative, 1.0),
            1.0 if c.kind == "hidden_relation" else 0.0,
            1.0 if c.kind == "anchor_fragment" else 0.0,
            1.0 if c.kind == "alignment_fragment" else 0.0,
            1.0 if c.kind == "missing_activity" else 0.0,
            1.0 if c.kind == "relaxed_transition" else 0.0,
            1.0 if c.kind in {"tau_prune", "arc_prune"} else 0.0,
            1.0 / (1.0 + max(complexity, 0)),
            complexity / max(max_complexity, 1),
            len(c.inserted) / 10.0,
            endpoint_overlap,
            1.0 if c.visible_duplicate_count == 0 else 0.0,
            1.0 if c.anchor is not None else 0.0,
        ])
    return torch.tensor(rows, dtype=torch.float32)

def deduplicate_candidates(candidates: Iterable[Candidate], used_keys: Set) -> List[Candidate]:
    best = {}
    for c in candidates:
        if c.key in used_keys:
            continue
        if c.key not in best or c.support > best[c.key].support:
            best[c.key] = c
    return list(best.values())

def action_sort_score(candidate: Candidate, trace_count: int, prefer_prune: bool) -> float:
    support = math.log1p(candidate.support) / max(math.log1p(max(trace_count, 1)), 1.0)
    hidden_bonus = 0.35 if candidate.kind == "hidden_relation" else 0.0
    alignment_bonus = 0.25 if candidate.kind == "alignment_fragment" else 0.0
    missing_bonus = 0.30 if candidate.kind == "missing_activity" else 0.0
    relaxation_bonus = 0.32 if candidate.kind == "relaxed_transition" else 0.0
    anchor_bonus = 0.18 if candidate.kind == "anchor_fragment" else 0.0
    prune_bonus = 0.35 if prefer_prune and candidate.kind in {"tau_prune", "arc_prune"} else 0.0
    complexity_penalty = 0.035 * max(candidate.estimated_complexity, 0)
    duplicate_penalty = 0.08 * candidate.visible_duplicate_count
    return (support + hidden_bonus + alignment_bonus + missing_bonus + relaxation_bonus
            + anchor_bonus + prune_bonus - complexity_penalty - duplicate_penalty)

def diverse_candidate_indices(
    ranked_indices: Sequence[int],
    candidates: Sequence[Candidate],
    limit: int,
    per_relation_limit: int,
) -> List[int]:
    selected: List[int] = []
    relation_counts: Dict[Tuple[str, str, str, Tuple[str, ...]], int] = defaultdict(int)
    for idx in ranked_indices:
        cand = candidates[idx]
        relation_key = (cand.kind, cand.source_label, cand.target_label, cand.inserted)
        if relation_counts[relation_key] >= per_relation_limit:
            continue
        selected.append(idx)
        relation_counts[relation_key] += 1
        if len(selected) >= limit:
            return selected
    for idx in ranked_indices:
        if idx not in selected:
            selected.append(idx)
            if len(selected) >= limit:
                break
    return selected

def stratified_action_candidates(
    candidates: List[Candidate],
    trace_count: int,
    prefer_prune: bool,
    limit: int,
) -> List[Candidate]:
    if len(candidates) <= limit:
        return sorted(
            candidates,
            key=lambda c: action_sort_score(c, trace_count, prefer_prune),
            reverse=True,
        )

    buckets: Dict[str, List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate.kind].append(candidate)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda c: action_sort_score(c, trace_count, prefer_prune),
            reverse=True,
        )

    if prefer_prune:
        quotas = {
            "missing_activity": max(1, round(limit * 0.10)),
            "alignment_fragment": max(1, round(limit * 0.15)),
            "anchor_fragment": max(1, round(limit * 0.10)),
            "relaxed_transition": max(1, round(limit * 0.15)),
            "hidden_relation": max(1, round(limit * 0.20)),
            "tau_prune": max(1, round(limit * 0.20)),
            "arc_prune": max(1, round(limit * 0.10)),
        }
    else:
        quotas = {
            "missing_activity": max(1, round(limit * 0.10)),
            "alignment_fragment": max(1, round(limit * 0.15)),
            "anchor_fragment": max(1, round(limit * 0.10)),
            "relaxed_transition": max(1, round(limit * 0.20)),
            "hidden_relation": max(1, round(limit * 0.30)),
            "tau_prune": max(1, round(limit * 0.10)),
            "arc_prune": max(1, round(limit * 0.05)),
        }
    selected: List[Candidate] = []
    selected_keys: Set[Tuple[str, str, Tuple[str, ...], str, Optional[str], Optional[str]]] = set()
    relation_counts: Counter = Counter()

    for kind in ("missing_activity", "alignment_fragment", "anchor_fragment", "relaxed_transition",
                 "hidden_relation", "tau_prune", "arc_prune"):
        quota = quotas.get(kind, 0)
        if quota <= 0:
            continue
        added_for_kind = 0
        for candidate in buckets.get(kind, []):
            relation_key = (candidate.kind, candidate.source_label, candidate.target_label, candidate.inserted)
            per_relation_limit = 2 if candidate.kind == "hidden_relation" else 1
            if relation_counts[relation_key] >= per_relation_limit:
                continue
            if candidate.key not in selected_keys and len(selected) < limit:
                selected.append(candidate)
                selected_keys.add(candidate.key)
                relation_counts[relation_key] += 1
                added_for_kind += 1
            if added_for_kind >= quota:
                break

    remainder = sorted(
        [candidate for candidate in candidates if candidate.key not in selected_keys],
        key=lambda c: action_sort_score(c, trace_count, prefer_prune),
        reverse=True,
    )
    for candidate in remainder:
        if len(selected) >= limit:
            break
        selected.append(candidate)
        selected_keys.add(candidate.key)
    return selected

def candidate_already_in_model(candidate: Candidate, net: PetriNet) -> bool:
    if candidate.kind == "relaxed_transition":
        transitions = {transition.name: transition for transition in net.transitions}
        source = transitions.get(candidate.source_transition_name or "")
        if source is None:
            return False
        expected_inputs = {candidate.source_place_name}
        expected_outputs = {
            arc.target.name for arc in source.out_arcs
            if isinstance(arc.target, PetriNet.Place)
        }
        for transition in net.transitions:
            if transition is source or transition.label != source.label:
                continue
            actual_inputs = {
                arc.source.name for arc in transition.in_arcs
                if isinstance(arc.source, PetriNet.Place)
            }
            actual_outputs = {
                arc.target.name for arc in transition.out_arcs
                if isinstance(arc.target, PetriNet.Place)
            }
            if actual_inputs == expected_inputs and actual_outputs == expected_outputs:
                return True
        return False

    if candidate.kind == "anchor_fragment":
        if candidate.anchor is None:
            return False
        places = {p.name: p for p in net.places}
        anchor_place = places.get(candidate.anchor.place_name)
        if anchor_place is None:
            return False
        transitions = {t.name: t for t in net.transitions}
        target_trans = transitions.get(candidate.anchor.target_name)
        if target_trans is None:
            return False
        
        def dfs(place, idx):
            if idx == len(candidate.inserted):
                for arc in place.out_arcs:
                    if isinstance(arc.target, PetriNet.Transition) and arc.target is target_trans:
                        return True
                return False
            label = candidate.inserted[idx]
            for arc in place.out_arcs:
                if isinstance(arc.target, PetriNet.Transition) and arc.target.label == label:
                    trans = arc.target
                    for out_arc in trans.out_arcs:
                        if isinstance(out_arc.target, PetriNet.Place):
                            if dfs(out_arc.target, idx + 1):
                                return True
            return False
        return dfs(anchor_place, 0)
    
    elif candidate.kind in {"alignment_fragment", "missing_activity"}:
        source_label = candidate.source_label
        target_label = candidate.target_label
        inserted = candidate.inserted
        src_transitions = [t for t in net.transitions if t.label == source_label]
        if not src_transitions:
            return False
        tgt_transitions = [t for t in net.transitions if t.label == target_label]
        if not tgt_transitions:
            return False
        
        def dfs_seq(place, idx):
            if idx == len(inserted):
                for arc in place.out_arcs:
                    if isinstance(arc.target, PetriNet.Transition) and arc.target in tgt_transitions:
                        return True
                return False
            label = inserted[idx]
            for arc in place.out_arcs:
                if isinstance(arc.target, PetriNet.Transition) and arc.target.label == label:
                    trans = arc.target
                    for out_arc in trans.out_arcs:
                        if isinstance(out_arc.target, PetriNet.Place):
                            if dfs_seq(out_arc.target, idx + 1):
                                return True
            return False
        
        for src in src_transitions:
            for arc in src.out_arcs:
                if isinstance(arc.target, PetriNet.Place):
                    if dfs_seq(arc.target, 0):
                        return True
        return False
    
    elif candidate.kind == "hidden_relation":
        if candidate.source_place_name and candidate.target_place_name:
            places = {p.name: p for p in net.places}
            source_place = places.get(candidate.source_place_name)
            target_place = places.get(candidate.target_place_name)
            if source_place is None or target_place is None:
                return False
            for arc_out_place in source_place.out_arcs:
                if isinstance(arc_out_place.target, PetriNet.Transition) and arc_out_place.target.label is None:
                    tau = arc_out_place.target
                    for tau_out in tau.out_arcs:
                        if tau_out.target is target_place:
                            return True
            return False
        source_label = candidate.source_label
        target_label = candidate.target_label
        src_transitions = [t for t in net.transitions if t.label == source_label]
        tgt_transitions = [t for t in net.transitions if t.label == target_label]
        if not src_transitions or not tgt_transitions:
            return False
        for src in src_transitions:
            for arc_out_src in src.out_arcs:
                if isinstance(arc_out_src.target, PetriNet.Place):
                    place = arc_out_src.target
                    for arc_out_place in place.out_arcs:
                        if isinstance(arc_out_place.target, PetriNet.Transition) and arc_out_place.target.label is None:
                            tau = arc_out_place.target
                            for tau_out in tau.out_arcs:
                                if isinstance(tau_out.target, PetriNet.Place):
                                    for tgt_arc in tau_out.target.out_arcs:
                                        if isinstance(tgt_arc.target, PetriNet.Transition) and tgt_arc.target in tgt_transitions:
                                            return True
        return False
    
    elif candidate.kind == "tau_prune":
        trans_name = candidate.remove_transition_name
        return any(t.name == trans_name for t in net.transitions)
    
    elif candidate.kind == "arc_prune":
        for src_name, tgt_name in candidate.remove_arc_names:
            for arc in net.arcs:
                if arc.source.name == src_name and arc.target.name == tgt_name:
                    return True
        return False
    
    return False

def mine_dynamic_candidates_improved(variants, alignment_log, net, im, fm, trace_count, used_keys: Dict, args, prefer_prune=False):
    model_labels = visible_labels(net)
    dfg = log_directly_follows(variants, model_labels)
    footprint = model_visible_footprint(net)
    rel_base_cands = mine_hidden_relation_candidates(dfg, footprint, model_labels, args.min_support,
                                                     args.min_relative_support, trace_count, args.relation_candidate_limit)
    rel_cands = expand_hidden_relation_places(net, rel_base_cands, args.relation_candidate_limit * 6)
    anchors = extract_direct_visible_anchors(net)
    frag_cands = mine_anchor_fragment_candidates(variants, anchors, net, model_labels,
                                                 args.max_insert_len, args.min_support, args.fragment_candidate_limit)
    align_cands = mine_alignment_fragment_candidates(alignment_log, net, im, fm, model_labels,
                                                     args.min_support, args.max_insert_len, args.alignment_candidate_limit)
    relax_cands = mine_relaxed_transition_candidates(
        variants, net, args.min_support, args.relaxation_candidate_limit)
    prune_cands = mine_prune_candidates(net, args.prune_candidate_limit)
    all_cands = rel_cands + frag_cands + align_cands + relax_cands + prune_cands

    def candidate_is_applicable(candidate: Candidate) -> bool:
        if candidate.kind in {"tau_prune", "arc_prune"}:
            return candidate_already_in_model(candidate, net)
        return not candidate_already_in_model(candidate, net)

    max_retry = getattr(args, 'candidate_max_retry', 2)
    filtered = []
    for c in all_cands:
        if used_keys.get(c.key, 0) < max_retry and candidate_is_applicable(c):
            filtered.append(c)
    filtered = deduplicate_candidates(filtered, set())
    filtered = stratified_action_candidates(filtered, trace_count, prefer_prune, args.action_candidate_limit)
    return CandidateBatch(rel_cands, frag_cands, align_cands, relax_cands, prune_cands, filtered)