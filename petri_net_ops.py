import copy
from typing import Dict, List, Optional, Set, Tuple

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from utils import Candidate, Anchor, ADDED_PREFIXES


def visible_labels(net: PetriNet) -> Set[str]:
    return {t.label for t in net.transitions if t.label is not None}

def is_visible_transition(node: object) -> bool:
    return isinstance(node, PetriNet.Transition) and getattr(node, "label", None) is not None

def infer_final_marking(net: PetriNet, final_marking: Optional[Marking]) -> Marking:
    if final_marking:
        return final_marking
    sink_places = [p for p in net.places if not p.out_arcs]
    if not sink_places:
        sink_places = [p for p in net.places if "end" in p.name.lower() or "sink" in p.name.lower()]
    if not sink_places:
        raise ValueError("Cannot infer final marking: no sink place found.")
    return Marking({sorted(sink_places, key=lambda p: p.name)[0]: 1})

def transitions_by_label(net: PetriNet, label: str) -> List[PetriNet.Transition]:
    return sorted([t for t in net.transitions if t.label == label],
                  key=lambda t: (len(t.in_arcs) + len(t.out_arcs), t.name))

def extract_direct_visible_anchors(net: PetriNet) -> Dict[Tuple[str, str], List[Anchor]]:
    anchors = defaultdict(list)
    for place in net.places:
        incoming = [arc.source for arc in place.in_arcs if is_visible_transition(arc.source)]
        outgoing = [arc.target for arc in place.out_arcs if is_visible_transition(arc.target)]
        for source in incoming:
            for target in outgoing:
                anchors[(source.label, target.label)].append(
                    Anchor(source.label, target.label, source.name, target.name, place.name)
                )
    return anchors

def choose_anchor(anchors: Sequence[Anchor], net: PetriNet) -> Anchor:
    places = {p.name: p for p in net.places}
    transitions = {t.name: t for t in net.transitions}
    def cost(anchor: Anchor) -> Tuple[int, str, str, str]:
        place = places[anchor.place_name]
        source = transitions[anchor.source_name]
        target = transitions[anchor.target_name]
        return (len(place.in_arcs) + len(place.out_arcs), place.name, source.name, target.name)
    return sorted(anchors, key=cost)[0]

def model_visible_footprint(net: PetriNet, max_silent_depth: int = 10) -> Set[Tuple[str, str]]:
    footprint = set()
    visible = [t for t in net.transitions if t.label is not None]
    for source in visible:
        queue = [(arc.target, 0) for arc in source.out_arcs]
        visited = set()
        while queue:
            node, depth = queue.pop(0)
            node_name = getattr(node, "name", str(id(node)))
            state_key = (node_name, depth)
            if state_key in visited or depth > max_silent_depth:
                continue
            visited.add(state_key)
            if isinstance(node, PetriNet.Transition):
                if node.label is not None:
                    footprint.add((source.label, node.label))
                    continue
                queue.extend((arc.target, depth + 1) for arc in node.out_arcs)
            elif isinstance(node, PetriNet.Place):
                queue.extend((arc.target, depth) for arc in node.out_arcs)
    return footprint

def unique_name(existing: Set[str], prefix: str) -> str:
    idx = 1
    while True:
        name = f"{prefix}{idx}"
        if name not in existing:
            existing.add(name)
            return name
        idx += 1

def arc_exists(source: object, target: object) -> bool:
    return any(arc.target is target for arc in getattr(source, "out_arcs", []))

def add_arc_if_missing(source: object, target: object, net: PetriNet) -> Optional[PetriNet.Arc]:
    if arc_exists(source, target):
        return None
    return petri_utils.add_arc_from_to(source, target, net)

def clone_marking_for_net(marking: Marking, net: PetriNet) -> Marking:
    places = {p.name: p for p in net.places}
    return Marking({places[p.name]: v for p, v in marking.items() if p.name in places})

def collect_added_elements(net: PetriNet) -> Tuple[List[object], List[object], List[object]]:
    added_transitions = [t for t in net.transitions if any(t.name.startswith(prefix) for prefix in ADDED_PREFIXES)]
    added_places = [p for p in net.places if any(p.name.startswith(prefix) for prefix in ADDED_PREFIXES)]
    added_nodes = set(added_transitions) | set(added_places)
    added_arcs = [arc for arc in net.arcs if arc.source in added_nodes or arc.target in added_nodes]
    return added_transitions, added_places, added_arcs

def add_hidden_relation_route(net: PetriNet, candidate: Candidate, repair_index: int) -> Tuple[List[object], List[object], List[object]]:
    transitions = {t.name: t for t in net.transitions}
    places = {p.name: p for p in net.places}
    sources = ([transitions[candidate.source_transition_name]] if candidate.source_transition_name and candidate.source_transition_name in transitions
               else transitions_by_label(net, candidate.source_label))
    targets = ([transitions[candidate.target_transition_name]] if candidate.target_transition_name and candidate.target_transition_name in transitions
               else transitions_by_label(net, candidate.target_label))
    if not sources or not targets:
        raise ValueError(f"Cannot find endpoints for {candidate.action_label}")
    source = sources[0]
    target = targets[0]
    if candidate.source_place_name and candidate.target_place_name:
        source_post = [places[candidate.source_place_name]] if candidate.source_place_name in places else []
        target_pre = [places[candidate.target_place_name]] if candidate.target_place_name in places else []
    else:
        source_post = sorted([arc.target for arc in source.out_arcs if isinstance(arc.target, PetriNet.Place)],
                             key=lambda p: (len(p.out_arcs), p.name))
        target_pre = sorted([arc.source for arc in target.in_arcs if isinstance(arc.source, PetriNet.Place)],
                            key=lambda p: (len(p.in_arcs), p.name))
    if not source_post or not target_pre:
        raise ValueError(f"Endpoint places missing for {candidate.action_label}")
    existing_names = {t.name for t in net.transitions}
    transition = PetriNet.Transition(unique_name(existing_names, f"rl_rel_{repair_index}_tau_"), None)
    net.transitions.add(transition)
    added_arcs = []
    in_arc = add_arc_if_missing(source_post[0], transition, net)
    if in_arc is not None:
        added_arcs.append(in_arc)
    for place in target_pre:
        out_arc = add_arc_if_missing(transition, place, net)
        if out_arc is not None:
            added_arcs.append(out_arc)
    return [transition], [], added_arcs

def add_safe_duplicate_tau(net: PetriNet, candidate: Candidate, repair_index: int) -> Tuple[List[object], List[object], List[object]]:
    transitions = sorted([t for t in net.transitions if t.name == candidate.source_transition_name],
                         key=lambda t: t.name)
    if not transitions:
        raise ValueError(f"Transition not found for safe duplicate: {candidate.source_transition_name}")
    source = transitions[0]
    in_places = sorted([arc.source for arc in source.in_arcs if isinstance(arc.source, PetriNet.Place)], key=lambda p: p.name)
    out_places = sorted([arc.target for arc in source.out_arcs if isinstance(arc.target, PetriNet.Place)], key=lambda p: p.name)
    if not in_places or not out_places:
        raise ValueError(f"Endpoint places missing for safe duplicate: {source.name}")
    existing_names = {t.name for t in net.transitions}
    duplicate = PetriNet.Transition(unique_name(existing_names, f"rl_safe_{repair_index}_tau_"), None)
    net.transitions.add(duplicate)
    added_arcs = []
    for place in in_places:
        arc = add_arc_if_missing(place, duplicate, net)
        if arc is not None:
            added_arcs.append(arc)
    for place in out_places:
        arc = add_arc_if_missing(duplicate, place, net)
        if arc is not None:
            added_arcs.append(arc)
    if not added_arcs:
        raise ValueError("Safe duplicate produced no new arcs")
    return [duplicate], [], added_arcs

def add_relaxed_transition_duplicate(
    net: PetriNet,
    candidate: Candidate,
    repair_index: int,
) -> Tuple[List[object], List[object], List[object]]:
    transitions = {transition.name: transition for transition in net.transitions}
    places = {place.name: place for place in net.places}
    source = transitions.get(candidate.source_transition_name or "")
    input_place = places.get(candidate.source_place_name or "")
    if source is None or source.label is None or input_place is None:
        raise ValueError(f"Cannot find synchronization endpoint for {candidate.action_label}")
    if not any(arc.source is input_place for arc in source.in_arcs):
        raise ValueError(f"Input place is not connected to {source.name}")
    output_places = sorted(
        [arc.target for arc in source.out_arcs if isinstance(arc.target, PetriNet.Place)],
        key=lambda place: place.name,
    )
    if not output_places:
        raise ValueError(f"Transition has no output place: {source.name}")
    existing_names = {transition.name for transition in net.transitions}
    duplicate = PetriNet.Transition(
        unique_name(existing_names, f"rl_relax_{repair_index}_t_"), source.label)
    net.transitions.add(duplicate)
    added_arcs = [petri_utils.add_arc_from_to(input_place, duplicate, net)]
    for output_place in output_places:
        added_arcs.append(petri_utils.add_arc_from_to(duplicate, output_place, net))
    return [duplicate], [], added_arcs

def add_alignment_fragment_branch(net: PetriNet, candidate: Candidate, repair_index: int) -> Tuple[List[object], List[object], List[object]]:
    sources = transitions_by_label(net, candidate.source_label)
    targets = transitions_by_label(net, candidate.target_label)
    if not sources or not targets:
        raise ValueError(f"Cannot find endpoints for {candidate.action_label}")
    source = sources[0]
    target = targets[0]
    source_post = sorted([arc.target for arc in source.out_arcs if isinstance(arc.target, PetriNet.Place)],
                         key=lambda p: (len(p.out_arcs), p.name))
    target_pre = sorted([arc.source for arc in target.in_arcs if isinstance(arc.source, PetriNet.Place)],
                        key=lambda p: (len(p.in_arcs), p.name))
    if not source_post or not target_pre:
        raise ValueError(f"Endpoint places missing for {candidate.action_label}")
    existing_t_names = {t.name for t in net.transitions}
    existing_p_names = {p.name for p in net.places}
    added_transitions, added_places, added_arcs = [], [], []
    current_place = source_post[0]
    prefix = "rl_missing" if candidate.kind == "missing_activity" else "rl_align"
    for pos, label in enumerate(candidate.inserted, start=1):
        trans = PetriNet.Transition(unique_name(existing_t_names, f"{prefix}_{repair_index}_t{pos}_"), label)
        net.transitions.add(trans)
        added_transitions.append(trans)
        arc_in = add_arc_if_missing(current_place, trans, net)
        if arc_in:
            added_arcs.append(arc_in)
        next_place = PetriNet.Place(unique_name(existing_p_names, f"{prefix}_{repair_index}_p{pos}_"))
        net.places.add(next_place)
        added_places.append(next_place)
        added_arcs.append(petri_utils.add_arc_from_to(trans, next_place, net))
        current_place = next_place
    for place in target_pre:
        arc_to_target = add_arc_if_missing(current_place, target, net)
        if arc_to_target:
            added_arcs.append(arc_to_target)
        if place is not target_pre[0]:
            tau = PetriNet.Transition(unique_name(existing_t_names, f"{prefix}_{repair_index}_tau_"), None)
            net.transitions.add(tau)
            added_transitions.append(tau)
            added_arcs.append(petri_utils.add_arc_from_to(current_place, tau, net))
            added_arcs.append(petri_utils.add_arc_from_to(tau, place, net))
    return added_transitions, added_places, added_arcs

def add_anchor_fragment_branch(net: PetriNet, candidate: Candidate, repair_index: int) -> Tuple[List[object], List[object], List[object]]:
    if candidate.anchor is None:
        raise ValueError("Anchor fragment candidate without anchor.")
    places = {p.name: p for p in net.places}
    transitions = {t.name: t for t in net.transitions}
    anchor_place = places[candidate.anchor.place_name]
    target = transitions[candidate.anchor.target_name]
    existing_t_names = {t.name for t in net.transitions}
    existing_p_names = {p.name for p in net.places}
    added_transitions, added_places, added_arcs = [], [], []
    current_place = anchor_place
    for pos, label in enumerate(candidate.inserted, start=1):
        trans = PetriNet.Transition(unique_name(existing_t_names, f"rl_frag_{repair_index}_t{pos}_"), label)
        net.transitions.add(trans)
        added_transitions.append(trans)
        arc_in = add_arc_if_missing(current_place, trans, net)
        if arc_in:
            added_arcs.append(arc_in)
        next_place = PetriNet.Place(unique_name(existing_p_names, f"rl_frag_{repair_index}_p{pos}_"))
        net.places.add(next_place)
        added_places.append(next_place)
        added_arcs.append(petri_utils.add_arc_from_to(trans, next_place, net))
        current_place = next_place
    arc_to_target = add_arc_if_missing(current_place, target, net)
    if arc_to_target:
        added_arcs.append(arc_to_target)
    return added_transitions, added_places, added_arcs

def remove_tau_transition(net: PetriNet, candidate: Candidate) -> Tuple[List[object], List[object], List[object]]:
    if not candidate.remove_transition_name:
        raise ValueError("tau_prune missing transition name")
    transitions = {t.name: t for t in net.transitions}
    trans = transitions.get(candidate.remove_transition_name)
    if trans is None:
        raise ValueError(f"Transition not found: {candidate.remove_transition_name}")
    if trans.label is not None:
        raise ValueError(f"Refusing to remove visible transition: {trans.name}")
    removed_arcs = list(trans.in_arcs) + list(trans.out_arcs)
    petri_utils.remove_transition(net, trans)
    return [trans], [], removed_arcs

def remove_candidate_arcs(net: PetriNet, candidate: Candidate) -> Tuple[List[object], List[object], List[object]]:
    removed_arcs = []
    for src_name, tgt_name in candidate.remove_arc_names:
        for arc in list(net.arcs):
            if arc.source.name == src_name and arc.target.name == tgt_name:
                petri_utils.remove_arc(net, arc)
                removed_arcs.append(arc)
                break
    if not removed_arcs:
        raise ValueError(f"No arc found: {candidate.remove_arc_names}")
    return [], [], removed_arcs

def apply_candidate(net: PetriNet, candidate: Candidate, repair_index: int) -> Tuple[List[object], List[object], List[object], List[object], List[object], List[object]]:
    if candidate.kind == "safe_duplicate":
        added_t, added_p, added_a = add_safe_duplicate_tau(net, candidate, repair_index)
        return added_t, added_p, added_a, [], [], []
    if candidate.kind == "relaxed_transition":
        added_t, added_p, added_a = add_relaxed_transition_duplicate(net, candidate, repair_index)
        return added_t, added_p, added_a, [], [], []
    if candidate.kind == "hidden_relation":
        added_t, added_p, added_a = add_hidden_relation_route(net, candidate, repair_index)
        return added_t, added_p, added_a, [], [], []
    if candidate.kind == "anchor_fragment":
        added_t, added_p, added_a = add_anchor_fragment_branch(net, candidate, repair_index)
        return added_t, added_p, added_a, [], [], []
    if candidate.kind in {"alignment_fragment", "missing_activity"}:
        added_t, added_p, added_a = add_alignment_fragment_branch(net, candidate, repair_index)
        return added_t, added_p, added_a, [], [], []
    if candidate.kind == "tau_prune":
        removed_t, removed_p, removed_a = remove_tau_transition(net, candidate)
        return [], [], [], removed_t, removed_p, removed_a
    if candidate.kind == "arc_prune":
        removed_t, removed_p, removed_a = remove_candidate_arcs(net, candidate)
        return [], [], [], removed_t, removed_p, removed_a
    raise ValueError(f"Unknown candidate kind: {candidate.kind}")