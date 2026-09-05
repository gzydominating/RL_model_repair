import copy
import random
import math
from collections import defaultdict
from typing import List, Optional, Tuple

import pm4py
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import Marking, PetriNet

import torch

from utils import *
from candidate_mining import *
from petri_net_ops import *
from reinforcement_learning import *


def meaningful_repair_candidate(candidate: Candidate) -> bool:
    return candidate.kind != "safe_duplicate"

def f1_score(fitness: float, precision: float) -> float:
    if math.isnan(fitness) or math.isnan(precision) or fitness + precision <= 0:
        return float("nan")
    return 2.0 * fitness * precision / (fitness + precision)

def composite_score(metrics: Metrics) -> float:
    return 0.30 * metrics.fitness + 0.30 * metrics.precision + 0.25 * metrics.f1 + 0.15 * metrics.simplicity

def compute_quality_metrics(log: EventLog, net: PetriNet, initial_marking: Marking, final_marking: Marking) -> Metrics:
    fitness = replay_evaluator.apply(copy.deepcopy(log), net, initial_marking, final_marking)["average_trace_fitness"]
    precision = precision_evaluator.apply(copy.deepcopy(log), net, initial_marking, final_marking)
    simplicity = simplicity_evaluator.apply(net)
    f1 = f1_score(float(fitness), float(precision))
    metrics = Metrics(float(fitness), float(precision), float(simplicity), f1, 0.0)
    metrics.composite = composite_score(metrics)
    return metrics

def compute_transition_metrics(log: EventLog, net: PetriNet, initial_marking: Marking, final_marking: Marking,
                               current_metrics: Metrics, args) -> Metrics:
    if getattr(args, "candidate_eval_mode", "fast") != "fast":
        return compute_quality_metrics(log, net, initial_marking, final_marking)
    fitness = replay_evaluator.apply(copy.deepcopy(log), net, initial_marking, final_marking)["average_trace_fitness"]
    precision = current_metrics.precision
    simplicity = simplicity_evaluator.apply(net)
    f1 = f1_score(float(fitness), float(precision))
    metrics = Metrics(float(fitness), float(precision), float(simplicity), f1, 0.0)
    metrics.composite = composite_score(metrics)
    return metrics

def improved_reward(before: Metrics, after: Metrics, original: Metrics, args) -> float:
    if math.isnan(after.fitness) or math.isnan(after.precision):
        return -1.0
    fitness_delta = after.fitness - before.fitness
    precision_delta = after.precision - before.precision
    f1_delta = after.f1 - before.f1
    simplicity_delta = after.simplicity - before.simplicity
    comp_delta = after.composite - before.composite
    reward = (
        args.reward_composite_weight * comp_delta
        + args.reward_fitness_weight * fitness_delta
        + args.reward_precision_weight * precision_delta
        + args.reward_f1_weight * f1_delta
        + args.reward_simplicity_weight * simplicity_delta
    )
    if after.fitness < original.fitness - args.metric_tolerance:
        reward -= args.fitness_floor_penalty * (original.fitness - after.fitness)
    if fitness_delta >= -args.metric_tolerance and comp_delta > args.metric_tolerance:
        reward += args.progress_bonus
    if after.composite < original.composite - args.catastrophic_loss:
        reward -= args.catastrophic_penalty
    if after.fitness >= original.fitness - args.metric_tolerance and (comp_delta > 0 or f1_delta > 0):
        reward += args.fitness_floor_bonus
    return float(reward)

def violates_fitness_floor(metrics: Metrics, original: Metrics, args) -> bool:
    return metrics.fitness < original.fitness - args.metric_tolerance

def acceptable_transition(before: Metrics, after: Metrics, original: Metrics, args) -> bool:
    if violates_fitness_floor(after, original, args):
        return False
    if after.composite >= before.composite + args.accept_composite_epsilon:
        return True
    if after.f1 >= before.f1 + args.accept_f1_epsilon:
        return True
    if after.precision >= before.precision + args.accept_precision_epsilon and after.f1 >= before.f1 - args.max_f1_loss:
        return True
    if after.fitness >= before.fitness + args.accept_fitness_epsilon and after.composite >= before.composite - args.max_composite_loss:
        return True
    return False

def final_quality_ok(metrics: Metrics, original: Metrics, args) -> bool:
    if violates_fitness_floor(metrics, original, args):
        return False
    return metrics.composite >= original.composite - args.final_max_composite_loss - args.metric_tolerance

def structural_fallback_quality_ok(metrics: Metrics, original: Metrics, args) -> bool:
    if violates_fitness_floor(metrics, original, args):
        return False
    if metrics.precision < original.precision - args.max_precision_loss - args.metric_tolerance:
        return False
    if metrics.f1 < original.f1 - args.max_f1_loss - args.metric_tolerance:
        return False
    if metrics.simplicity < original.simplicity - args.max_simplicity_loss - args.metric_tolerance:
        return False
    return True

def outcome_label_score(label: Optional[str]) -> float:
    if not label:
        return 0.0
    upper = label.upper()
    outcome_terms = (
        "CANCEL", "DECLIN", "DENIED", "REJECT", "RETURN",
        "FINAL", "WITHDRAW", "REFUSE", "ABORT", "END"
    )
    return 1.0 if any(term in upper for term in outcome_terms) else 0.0

def evaluate_candidate_actions(current_net, current_im, current_fm, current_metrics, original_metrics,
                               candidates, log, repair_index, args) -> Tuple[List[float], List[Optional[Tuple]], List[str]]:
    rewards, trials, notes = [], [], []
    for cand in candidates:
        trial_net = copy.deepcopy(current_net)
        trial_im = clone_marking_for_net(current_im, trial_net)
        trial_fm = clone_marking_for_net(current_fm, trial_net)
        try:
            _, _, _, rem_t, rem_p, rem_a = apply_candidate(trial_net, cand, repair_index)
            trial_metrics = compute_transition_metrics(log, trial_net, trial_im, trial_fm, current_metrics, args)
            reward = improved_reward(current_metrics, trial_metrics, original_metrics, args)
            complexity_penalty = args.action_complexity_penalty * max(cand.estimated_complexity, 0)
            duplicate_penalty = args.action_duplicate_penalty * cand.visible_duplicate_count
            reward -= complexity_penalty + duplicate_penalty
            note = f"reward={reward:.6f}, d_comp={metric_delta(trial_metrics, current_metrics, 'composite'):.6f}"
            rewards.append(reward)
            trials.append((trial_net, trial_im, trial_fm, trial_metrics, rem_t, rem_p, rem_a))
            notes.append(note)
        except Exception as e:
            rewards.append(-1.0)
            trials.append(None)
            notes.append(f"invalid: {e}")

    if getattr(args, "candidate_eval_mode", "fast") == "fast" and candidates:
        validation_indices = set(staged_validation_indices(
            candidates, rewards, getattr(args, "precision_validation_trials", len(candidates))))
        for idx, trial in enumerate(trials):
            if trial is None:
                continue
            if idx not in validation_indices:
                rewards[idx] = -1.0
                trials[idx] = None
                notes[idx] = f"{notes[idx]}, screened out before precision validation"
                continue
            cand = candidates[idx]
            try:
                validated_metrics = compute_quality_metrics(log, trial[0], trial[1], trial[2])
                reward = improved_reward(current_metrics, validated_metrics, original_metrics, args)
                reward -= args.action_complexity_penalty * max(cand.estimated_complexity, 0)
                reward -= args.action_duplicate_penalty * cand.visible_duplicate_count
                rewards[idx] = reward
                trials[idx] = (trial[0], trial[1], trial[2], validated_metrics,
                               trial[4], trial[5], trial[6])
                notes[idx] = (
                    f"reward={reward:.6f}, "
                    f"d_comp={metric_delta(validated_metrics, current_metrics, 'composite'):.6f}, "
                    "precision-validated"
                )
            except Exception as exc:
                rewards[idx] = -1.0
                trials[idx] = None
                notes[idx] = f"precision validation failed: {exc}"
    return rewards, trials, notes

def full_validate_action_shortlist(
    candidates: Sequence[Candidate],
    rewards: List[float],
    trials: List[Optional[Tuple]],
    notes: List[str],
    sample_acceptable_actions: Sequence[int],
    current_full_metrics: Metrics,
    original_full_metrics: Metrics,
    full_log: EventLog,
    args,
) -> List[int]:
    if not sample_acceptable_actions:
        return []
    local_candidates = [candidates[idx] for idx in sample_acceptable_actions]
    local_rewards = [rewards[idx] for idx in sample_acceptable_actions]
    local_indices = staged_validation_indices(
        local_candidates, local_rewards, max(1, args.full_validation_top_k))
    shortlist = [sample_acceptable_actions[idx] for idx in local_indices]
    full_acceptable: List[int] = []
    for idx in shortlist:
        trial = trials[idx]
        if trial is None:
            continue
        cand = candidates[idx]
        try:
            full_metrics = compute_quality_metrics(full_log, trial[0], trial[1], trial[2])
            full_reward = improved_reward(
                current_full_metrics, full_metrics, original_full_metrics, args)
            full_reward -= args.action_complexity_penalty * max(cand.estimated_complexity, 0)
            full_reward -= args.action_duplicate_penalty * cand.visible_duplicate_count
            rewards[idx] = full_reward
            trials[idx] = (trial[0], trial[1], trial[2], full_metrics,
                           trial[4], trial[5], trial[6])
            notes[idx] = (
                f"reward={full_reward:.6f}, "
                f"d_comp={metric_delta(full_metrics, current_full_metrics, 'composite'):.6f}, "
                "full-log-validated"
            )
            if (final_quality_ok(full_metrics, original_full_metrics, args)
                    and acceptable_transition(
                        current_full_metrics, full_metrics, original_full_metrics, args)):
                full_acceptable.append(idx)
        except Exception as exc:
            rewards[idx] = -1.0
            trials[idx] = None
            notes[idx] = f"full-log validation failed: {exc}"
    return full_acceptable

def find_mandatory_safe_repair(base_net, initial_marking, final_marking, variants, trace_count,
                               decision_log, full_log, alignment_log, original_metrics, args):
    search_args = copy.copy(args)
    search_args.min_support = max(1, min(args.min_support, args.fallback_min_support))
    search_args.min_relative_support = min(args.min_relative_support, args.fallback_min_relative_support)
    search_args.action_candidate_limit = max(args.action_candidate_limit, args.fallback_action_candidate_limit)
    search_args.relation_candidate_limit = max(args.relation_candidate_limit, args.fallback_relation_candidate_limit)
    search_args.fragment_candidate_limit = max(args.fragment_candidate_limit, args.fallback_fragment_candidate_limit)
    search_args.alignment_candidate_limit = max(args.alignment_candidate_limit, args.fallback_alignment_candidate_limit)
    search_args.prune_candidate_limit = max(args.prune_candidate_limit, args.fallback_prune_candidate_limit)
    search_args.candidate_eval_mode = "full"
    used_keys = defaultdict(int)
    base_copy = copy.deepcopy(base_net)
    base_im = clone_marking_for_net(initial_marking, base_copy)
    base_fm = clone_marking_for_net(final_marking, base_copy)
    sample_original_metrics = compute_quality_metrics(decision_log, base_copy, base_im, base_fm)
    batch = mine_dynamic_candidates_improved(variants, alignment_log, base_copy, base_im, base_fm,
                                             trace_count, used_keys, search_args,
                                             prefer_prune=original_metrics.fitness >= args.fitness_near_one)
    if not batch.candidates:
        return find_structural_noop_repair(base_net, initial_marking, final_marking,
                                           full_log, original_metrics, args)
    prefer_prune = original_metrics.fitness >= args.fitness_near_one
    ranked_static = sorted(
        range(len(batch.candidates)),
        key=lambda i: fallback_static_rank(batch.candidates[i], trace_count, prefer_prune),
        reverse=True,
    )
    meaningful_static = [idx for idx in ranked_static if meaningful_repair_candidate(batch.candidates[idx])]
    static_pool = meaningful_static or ranked_static
    pool_limit = max(args.full_fallback_trials * args.fallback_relation_variant_factor,
                     args.fallback_sample_validation_trials)
    static_pool = diverse_candidate_indices(
        static_pool, batch.candidates, pool_limit, args.fallback_relation_variant_factor)
    sample_candidates = [batch.candidates[idx] for idx in static_pool]
    rewards, trials, notes = evaluate_candidate_actions(base_copy, base_im, base_fm, sample_original_metrics,
                                                        sample_original_metrics, sample_candidates, decision_log, 1, search_args)
    feasible_local = []
    for local_idx, trial in enumerate(trials):
        cand = sample_candidates[local_idx]
        if trial is None:
            continue
        metrics = trial[3]
        precision_floor = sample_original_metrics.precision - args.fallback_hard_sample_precision_loss - args.metric_tolerance
        f1_floor = sample_original_metrics.f1 - args.fallback_hard_sample_f1_loss - args.metric_tolerance
        if metrics.fitness < sample_original_metrics.fitness - args.metric_tolerance:
            continue
        if metrics.precision < precision_floor:
            continue
        if metrics.f1 < f1_floor:
            continue
        if metrics.composite >= sample_original_metrics.composite - args.max_composite_loss:
            feasible_local.append(local_idx)
        elif outcome_label_score(cand.target_label) > 0 and metrics.fitness >= sample_original_metrics.fitness - args.metric_tolerance:
            feasible_local.append(local_idx)
    if not feasible_local:
        return find_structural_noop_repair(base_net, initial_marking, final_marking,
                                           full_log, original_metrics, args)
    ranked = sorted(
        feasible_local,
        key=lambda i: fallback_sample_rank(trials[i][3], sample_original_metrics, rewards[i], sample_candidates[i], args),
        reverse=True,
    )
    best_full = None
    meaningful_ranked = [idx for idx in ranked if meaningful_repair_candidate(sample_candidates[idx])]
    if not meaningful_ranked:
        meaningful_ranked = ranked
    for chosen in meaningful_ranked[:args.full_fallback_trials]:
        trial = trials[chosen]
        cand = sample_candidates[chosen]
        full_metrics = compute_quality_metrics(full_log, trial[0], trial[1], trial[2])
        if not final_quality_ok(full_metrics, original_metrics, args):
            continue
        if full_metrics.composite >= original_metrics.composite + args.fallback_early_stop_composite_gain:
            decision = RepairDecision(1, "fallback-apply", cand, rewards[chosen],
                                      original_metrics, full_metrics, 1.0,
                                      f"{notes[chosen]}, mandatory safe repair/full-log checked/early-stop")
            return trial[0], trial[1], trial[2], full_metrics, [cand], batch, decision
        if best_full is None or full_metrics.composite > best_full[3].composite + args.metric_tolerance:
            best_full = (chosen, trial, cand, full_metrics)
    if best_full is None:
        return find_structural_noop_repair(base_net, initial_marking, final_marking,
                                           full_log, original_metrics, args)
    chosen, trial, cand, full_metrics = best_full
    decision = RepairDecision(1, "fallback-apply", cand, rewards[chosen],
                              original_metrics, full_metrics, 1.0,
                              f"{notes[chosen]}, mandatory safe repair/full-log checked")
    return trial[0], trial[1], trial[2], full_metrics, [cand], batch, decision

def find_structural_noop_repair(base_net, initial_marking, final_marking, full_log,
                                original_metrics, args):
    best_strict = None
    best_relaxed = None
    for trans in sorted(base_net.transitions, key=lambda t: (t.label is None, t.name)):
        if trans.label is None or not trans.in_arcs or not trans.out_arcs:
            continue
        trial_net = copy.deepcopy(base_net)
        trial_im = clone_marking_for_net(initial_marking, trial_net)
        trial_fm = clone_marking_for_net(final_marking, trial_net)
        cand = Candidate(kind="safe_duplicate", source_label=trans.label, target_label=trans.label,
                         support=1, source_transition_name=trans.name)
        try:
            apply_candidate(trial_net, cand, 1)
            metrics = compute_quality_metrics(full_log, trial_net, trial_im, trial_fm)
        except Exception:
            continue
        result = (trial_net, trial_im, trial_fm, metrics, cand)
        if final_quality_ok(metrics, original_metrics, args):
            if (best_strict is None
                    or metrics.composite > best_strict[3].composite + args.metric_tolerance):
                best_strict = result
            if metrics.composite >= original_metrics.composite + args.fallback_early_stop_composite_gain:
                break
        elif structural_fallback_quality_ok(metrics, original_metrics, args):
            if (best_relaxed is None
                    or metrics.composite > best_relaxed[3].composite + args.metric_tolerance):
                best_relaxed = result
    chosen = best_strict or best_relaxed
    if chosen is None:
        return None
    trial_net, trial_im, trial_fm, metrics, cand = chosen
    strict = best_strict is not None
    batch = CandidateBatch([], [], [], [], [], [cand])
    note = "structural non-empty repair/full-log checked"
    if not strict:
        note += "/fitness-preserving fallback"
    decision = RepairDecision(1, "fallback-safe-duplicate", cand,
                              metrics.composite - original_metrics.composite,
                              original_metrics, metrics, 1.0, note)
    return trial_net, trial_im, trial_fm, metrics, [cand], batch, decision

def selected_removed_elements(base_net: PetriNet, selected: Sequence[Candidate]) -> Tuple[List[object], List[object], List[object]]:
    transitions = {t.name: t for t in base_net.transitions}
    places = {p.name: p for p in base_net.places}
    arcs = {(a.source.name, a.target.name): a for a in base_net.arcs}
    removed_transitions = []
    removed_places = []
    removed_arcs = []
    for cand in selected:
        if cand.kind == "tau_prune" and cand.remove_transition_name in transitions:
            trans = transitions[cand.remove_transition_name]
            removed_transitions.append(trans)
            removed_arcs.extend(list(trans.in_arcs) + list(trans.out_arcs))
        elif cand.kind == "arc_prune":
            for arc_key in cand.remove_arc_names:
                if arc_key in arcs:
                    removed_arcs.append(arcs[arc_key])
    return removed_transitions, removed_places, removed_arcs

def sequential_lpmr_repair_improved(base_net, initial_marking, final_marking, variants, trace_count,
                                    full_log, decision_log, alignment_log, original_metrics, args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    decision_original_metrics = compute_quality_metrics(
        decision_log, base_net, initial_marking, final_marking)

    current_net = copy.deepcopy(base_net)
    current_im = clone_marking_for_net(initial_marking, current_net)
    current_fm = clone_marking_for_net(final_marking, current_net)
    current_metrics = decision_original_metrics
    current_full_metrics = original_metrics
    best_net = copy.deepcopy(current_net)
    best_im = clone_marking_for_net(current_im, best_net)
    best_fm = clone_marking_for_net(current_fm, best_net)
    best_metrics = original_metrics
    best_selected: List[Candidate] = []

    policy = AttentionActorCritic(7, 14, args.hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.ppo_lr)
    buffer = MDPRolloutBuffer(capacity=args.buffer_capacity)

    selected: List[Candidate] = []
    decisions: List[RepairDecision] = []
    recent_rewards: List[float] = []
    used_keys = defaultdict(int)
    last_batch = CandidateBatch([], [], [], [], [], [])
    no_progress_steps = 0
    stop_votes = 0

    for step in range(1, args.max_steps + 1):
        if len(selected) >= args.max_repairs:
            decisions.append(RepairDecision(step, "stop", None, 0.0, current_metrics, current_metrics, 1.0, "max repairs reached"))
            break

        prefer_prune = current_metrics.fitness >= args.fitness_near_one
        last_batch = mine_dynamic_candidates_improved(variants, alignment_log, current_net, current_im, current_fm,
                                                      trace_count, used_keys, args, prefer_prune)
        candidates = last_batch.candidates
        if not candidates:
            decisions.append(RepairDecision(step, "stop", None, 0.0, current_metrics, current_metrics, 1.0, "no candidates"))
            break

        state = state_tensor(current_metrics, current_net, len(candidates), recent_rewards)
        cand_feats = feature_tensor(candidates, trace_count)
        decision_current_metrics = compute_quality_metrics(
            decision_log, current_net, current_im, current_fm)
        rewards, trials, notes = evaluate_candidate_actions(current_net, current_im, current_fm,
                                                            decision_current_metrics, decision_original_metrics,
                                                            candidates, decision_log, len(selected) + 1, args)
        sample_acceptable_actions = [
            idx for idx, trial in enumerate(trials)
            if trial is not None and acceptable_transition(
                decision_current_metrics, trial[3], decision_original_metrics, args)
        ]
        full_acceptable_actions = full_validate_action_shortlist(
            candidates, rewards, trials, notes, sample_acceptable_actions,
            current_full_metrics, original_metrics, full_log, args)
        valid_actions = full_acceptable_actions if args.mask_unacceptable_actions else [
            idx for idx in full_acceptable_actions if trials[idx] is not None
        ]
        force_action = step <= args.force_explore_steps or not selected
        if not valid_actions:
            decisions.append(RepairDecision(step, "stop", None, 0.0, current_metrics, current_metrics,
                                            1.0, "no action satisfies fitness floor"))
            break

        action_idx, probability, old_log_prob, value, _ = choose_mdp_action(
            policy, state, cand_feats, valid_actions, force_action, rewards, args)

        if action_idx >= len(candidates):
            stop_votes += 1
            done = stop_votes >= args.stop_patience
            with torch.no_grad():
                next_value = torch.tensor(0.0)
            buffer.add(MDPTransition(state.clone(), cand_feats.clone(), action_idx,
                                     args.stop_reward, done, old_log_prob.detach(), value.detach(), next_value))
            ppo_update_from_buffer(policy, optimizer, buffer, args)
            decisions.append(RepairDecision(step, "stop", None, args.stop_reward,
                                            current_metrics, current_metrics, probability, "policy stop"))
            if done:
                break
            continue

        stop_votes = 0
        cand = candidates[action_idx]
        trial = trials[action_idx]
        reward = rewards[action_idx]
        note = notes[action_idx]
        used_keys[cand.key] += 1

        if trial is None:
            done = False
            next_value = value.detach()
            penalty = min(reward, -args.invalid_action_penalty)
            buffer.add(MDPTransition(state.clone(), cand_feats.clone(), action_idx,
                                     penalty, done, old_log_prob.detach(), value.detach(), next_value))
            ppo_update_from_buffer(policy, optimizer, buffer, args)
            decisions.append(RepairDecision(step, "invalid", cand, penalty,
                                            current_metrics, current_metrics, probability, note))
            no_progress_steps += 1
        elif not acceptable_transition(current_full_metrics, trial[3], original_metrics, args):
            penalty = min(reward, -args.unacceptable_action_penalty)
            done = False
            next_value = value.detach()
            if violates_fitness_floor(trial[3], original_metrics, args):
                note = f"{note}, rejected: below original fitness"
            else:
                note = f"{note}, rejected: no accepted quality gain"
            buffer.add(MDPTransition(state.clone(), cand_feats.clone(), action_idx,
                                     penalty, done, old_log_prob.detach(), value.detach(), next_value))
            ppo_update_from_buffer(policy, optimizer, buffer, args)
            decisions.append(RepairDecision(step, "reject", cand, penalty,
                                            current_metrics, current_metrics, probability, note))
            no_progress_steps += 1
        else:
            before = current_full_metrics
            current_net, current_im, current_fm, current_metrics, _, _, _ = trial
            current_full_metrics = current_metrics
            recent_rewards.append(reward)
            selected.append(cand)
            next_prefer_prune = current_metrics.fitness >= args.fitness_near_one
            next_batch = mine_dynamic_candidates_improved(variants, alignment_log, current_net, current_im, current_fm,
                                                          trace_count, used_keys, args, next_prefer_prune)
            next_state = state_tensor(current_metrics, current_net, len(next_batch.candidates), recent_rewards)
            next_feats = feature_tensor(next_batch.candidates, trace_count)
            done = len(selected) >= args.max_repairs or step >= args.max_steps
            with torch.no_grad():
                _, next_value_tensor = policy(next_state, next_feats)
            buffer.add(MDPTransition(state.clone(), cand_feats.clone(), action_idx,
                                     reward, done, old_log_prob.detach(), value.detach(), next_value_tensor.detach().view(())))
            ppo_update_from_buffer(policy, optimizer, buffer, args)
            decisions.append(RepairDecision(step, "apply", cand, reward,
                                            before, current_metrics, probability, note))

            if (current_metrics.fitness >= original_metrics.fitness - args.metric_tolerance
                    and current_metrics.composite > best_metrics.composite + args.metric_tolerance
                    and selected):
                best_net = copy.deepcopy(current_net)
                best_im = clone_marking_for_net(current_im, best_net)
                best_fm = clone_marking_for_net(current_fm, best_net)
                best_metrics = current_metrics
                best_selected = list(selected)
                no_progress_steps = 0
            else:
                no_progress_steps += 1

        if no_progress_steps >= args.no_improvement_patience:
            break

    if best_selected and best_metrics.fitness >= original_metrics.fitness - args.metric_tolerance:
        current_net, current_im, current_fm, current_metrics = best_net, best_im, best_fm, best_metrics
        selected = best_selected
        decisions.append(RepairDecision(len(decisions) + 1, "select-best", None,
                                        best_metrics.composite - original_metrics.composite,
                                        current_metrics, current_metrics, 1.0, "best feasible MDP state"))

    if selected:
        sample_metrics = current_metrics
        full_metrics = compute_quality_metrics(full_log, current_net, current_im, current_fm)
        decisions.append(RepairDecision(len(decisions) + 1, "full-log-check", None,
                                        full_metrics.composite - original_metrics.composite,
                                        sample_metrics, full_metrics, 1.0, "full log validation"))
        current_metrics = full_metrics
        if not final_quality_ok(current_metrics, original_metrics, args):
            selected = []
            current_net = copy.deepcopy(base_net)
            current_im = clone_marking_for_net(initial_marking, current_net)
            current_fm = clone_marking_for_net(final_marking, current_net)
            current_metrics = original_metrics
            decisions.append(RepairDecision(len(decisions) + 1, "reject-full-log", None,
                                            -args.unacceptable_action_penalty,
                                            full_metrics, original_metrics, 1.0,
                                            "full-log metrics failed final quality floor"))

    if not selected or violates_fitness_floor(current_metrics, original_metrics, args):
        fallback = find_mandatory_safe_repair(base_net, initial_marking, final_marking, variants, trace_count,
                                              decision_log, full_log, alignment_log, original_metrics, args)
        if fallback is not None:
            current_net, current_im, current_fm, current_metrics, selected, fallback_batch, fallback_decision = fallback
            last_batch = fallback_batch
            decisions.append(fallback_decision)
        else:
            current_net = copy.deepcopy(base_net)
            current_im = clone_marking_for_net(initial_marking, current_net)
            current_fm = clone_marking_for_net(final_marking, current_net)
            current_metrics = original_metrics
            selected = []
            decisions.append(RepairDecision(len(decisions) + 1, "failed-safe-repair", None, -1.0,
                                            current_metrics, current_metrics, 1.0,
                                            "no fitness-preserving modification found"))
    removed_transitions, removed_places, removed_arcs = selected_removed_elements(base_net, selected)
    return (current_net, current_im, current_fm, current_metrics, selected, decisions,
            removed_transitions, removed_places, removed_arcs, last_batch)