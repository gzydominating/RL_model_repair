import torch
from typing import Sequence, Optional, List

from utils import MDPTransition

class AttentionActorCritic(torch.nn.Module):
    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int):
        super().__init__()
        self.state_proj = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.state_attention = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_attention = torch.nn.Linear(candidate_dim, hidden_dim, bias=False)
        self.attention_vector = torch.nn.Linear(hidden_dim, 1)
        self.stop_head = torch.nn.Linear(hidden_dim, 1)
        self.value_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, candidate_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state_hidden = self.state_proj(state)
        value = self.value_head(state_hidden).view(-1)
        stop_logit = self.stop_head(state_hidden).view(1)
        if candidate_features.numel() == 0:
            return stop_logit, value
        state_term = self.state_attention(state_hidden).expand(candidate_features.shape[0], -1)
        candidate_term = self.candidate_attention(candidate_features)
        candidate_logits = self.attention_vector(torch.tanh(state_term + candidate_term)).view(-1)
        return torch.cat([candidate_logits, stop_logit]), value

class MDPRolloutBuffer:
    def __init__(self, capacity: int = 128):
        self.transitions: List[MDPTransition] = []
        self.capacity = capacity

    def add(self, transition: MDPTransition) -> None:
        self.transitions.append(transition)
        if len(self.transitions) > self.capacity:
            self.transitions.pop(0)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

def ppo_update_from_buffer(policy: AttentionActorCritic, optimizer: torch.optim.Optimizer,
                           buffer: MDPRolloutBuffer, args) -> None:
    if len(buffer) == 0:
        return
    transitions = list(buffer.transitions)
    batch_size = len(transitions)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
    dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)
    old_values = torch.stack([t.value.detach().view(()) for t in transitions])
    next_values = torch.stack([t.next_value.detach().view(()) for t in transitions])
    old_log_probs = torch.stack([t.old_log_prob.detach().view(()) for t in transitions])
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long)

    advantages = torch.zeros(batch_size, dtype=torch.float32)
    gae = torch.tensor(0.0)
    for i in reversed(range(batch_size)):
        bootstrap = next_values[i] * (1.0 - dones[i])
        delta = rewards[i] + args.gamma * bootstrap - old_values[i]
        gae = delta + args.gamma * args.gae_lambda * (1.0 - dones[i]) * gae
        advantages[i] = gae
    returns = advantages + old_values
    if advantages.numel() > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(args.ppo_epochs):
        new_log_probs = []
        new_values = []
        entropies = []
        for i, transition in enumerate(transitions):
            logits, val = policy(transition.state, transition.candidate_features)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs.append(dist.log_prob(actions[i]))
            new_values.append(val.view(()))
            entropies.append(dist.entropy())
        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.stack(new_values)
        entropy = torch.stack(entropies).mean()
        ratios = torch.exp(new_log_probs - old_log_probs.detach())
        clipped_ratios = torch.clamp(ratios, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip)
        actor_loss = -torch.min(ratios * advantages.detach(), clipped_ratios * advantages.detach()).mean()
        critic_loss = torch.nn.functional.mse_loss(new_values, returns.detach())
        loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
    if args.clear_rollout_after_update:
        buffer.clear()

def masked_action_distribution(
    policy: AttentionActorCritic,
    state: torch.Tensor,
    cand_feats: torch.Tensor,
    valid_actions: Sequence[int],
    force_action: bool,
    reward_bias: Optional[Sequence[float]],
    args,
) -> Tuple[torch.distributions.Categorical, torch.Tensor, torch.Tensor]:
    logits, value = policy(state, cand_feats)
    masked_logits = logits.clone()
    allowed = set(valid_actions)
    if not force_action:
        allowed.add(len(logits) - 1)
    for idx in range(len(masked_logits)):
        if idx not in allowed:
            masked_logits[idx] = -1e9
    if reward_bias is not None and args.reward_guided_selection_weight > 0:
        bias = torch.zeros_like(masked_logits)
        reward_tensor = torch.tensor(list(reward_bias), dtype=torch.float32)
        if reward_tensor.numel() > 1 and reward_tensor.std() > 1e-8:
            reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)
        elif reward_tensor.numel() > 0:
            reward_tensor = reward_tensor - reward_tensor.mean()
        bias[:len(reward_tensor)] = reward_tensor
        masked_logits = masked_logits + args.reward_guided_selection_weight * bias
    return torch.distributions.Categorical(logits=masked_logits), logits, value

def choose_mdp_action(
    policy: AttentionActorCritic,
    state: torch.Tensor,
    cand_feats: torch.Tensor,
    valid_actions: Sequence[int],
    force_action: bool,
    reward_bias: Sequence[float],
    args,
) -> Tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    dist, logits, value = masked_action_distribution(
        policy, state, cand_feats, valid_actions, force_action, reward_bias, args)
    if args.greedy_accepted_action and valid_actions:
        action = max(valid_actions, key=lambda idx: reward_bias[idx] if idx < len(reward_bias) else -float("inf"))
    elif args.deterministic_policy:
        action = int(torch.argmax(dist.logits).item())
    else:
        action = int(dist.sample().item())
    log_prob = dist.log_prob(torch.tensor(action))
    probability = float(torch.softmax(dist.logits, dim=0)[action].detach().cpu().item())
    return action, probability, log_prob, value.view(()), logits.detach()

def state_tensor(metrics, net, num_candidates, recent_rewards) -> torch.Tensor:
    hist = sum(recent_rewards[-5:]) / max(len(recent_rewards[-5:]), 1)
    state = [
        metrics.fitness,
        metrics.precision,
        metrics.f1,
        metrics.simplicity,
        min((len(net.transitions) + len(net.places)) / 250.0, 4.0),
        min(num_candidates / 50.0, 4.0),
        max(min(hist, 1.0), -1.0),
    ]
    return torch.tensor(state, dtype=torch.float32)