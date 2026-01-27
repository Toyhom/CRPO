# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int) -> None:
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class EntropyAwareRoleKLController(KLController):
    """
    Core CRPO innovation component: entropy-aware role-based adaptive multi-channel KL controller.
    
    Innovations:
    1. Multi-Channel: Independently maintain KL coefficients for each role.
    2. Entropy-Scaling: Dynamically scale Target KL based on the predicted entropy of the Reference Model for that role.
       Roles with higher entropy are not only "hard to predict" but also have "high style tolerance", so they are given a higher Target KL.
    """

    def __init__(self, init_kl_coef: float, base_target_kl: float, horizon: float):
        self.kl_coefs: Dict[str, float] = defaultdict(lambda: init_kl_coef)
        self.base_target_kl = base_target_kl
        self.horizon = horizon
        
        # Record the historical average entropy of each role
        self.role_entropy_ma: Dict[str, float] = defaultdict(lambda: 0.0)
        self.global_avg_entropy = None
        self.alpha_ma = 0.05 # Moving average coefficient
        self.role_scale_factor: Dict[str, float] = defaultdict(lambda: 0.0)

    def update(self, current_kl: float, n_steps: int, role_name: str = None, current_entropy: float = None) -> None:
        """
        Args:
            current_kl: Average KL of the role in the current Batch
            n_steps: Update steps
            role_name: Role name
            current_entropy: Average entropy of Ref Model for the role in the current Batch (Shannon Entropy)
        """
        if role_name is None:
            return

        # 1. Update entropy statistics for the role (EMA)
        if current_entropy is not None:
            old_ma = self.role_entropy_ma[role_name]
            # Assign directly if encountered for the first time
            if old_ma == 0.0:
                self.role_entropy_ma[role_name] = current_entropy
            else:
                self.role_entropy_ma[role_name] = (1 - self.alpha_ma) * old_ma + self.alpha_ma * current_entropy

            if self.global_avg_entropy is None:
                self.global_avg_entropy = current_entropy
            else:
                self.global_avg_entropy = (1 - 0.01) * self.global_avg_entropy + 0.01 * current_entropy


        role_entropy = self.role_entropy_ma[role_name]
        
        # Scaling factor: relative entropy value
        # Limit scaling between [0.5, 2.0] to prevent extreme cases

        scale_factor = np.clip(np.power(self.global_avg_entropy / (role_entropy + 1e-6), 1.5), 0.5, 2.0)

        self.role_scale_factor[role_name] = scale_factor
        
        adaptive_target = self.base_target_kl * scale_factor

        # 3. Standard PID/Adaptive control logic (P-Controller)
        current_coef = self.kl_coefs[role_name]
        proportional_error = np.clip(current_kl / adaptive_target - 1, -0.2, 0.2)
        mult = 1 - proportional_error * n_steps / self.horizon
        
        self.kl_coefs[role_name] = current_coef * mult


    def get_coef(self, role_name: str) -> float:
        return self.kl_coefs[role_name]

    def get_scale_factor(self, role_name: str) -> float:
        return self.role_scale_factor[role_name]


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    elif algorithm_config.kl_type == "entropy_aware_role":
        kl_ctrl = EntropyAwareRoleKLController(
            init_kl_coef=algorithm_config.kl_coef,
            base_target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns














@torch.no_grad()
def compute_drgrpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = scores[i] - id2mean[index[i]]

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns





@torch.no_grad()
def compute_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GDPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, num_rewards)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,) Prompt IDs to group by.

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    # 1. Aggregate rewards across the sequence length
    # shape: (bs, num_rewards)

    bsz, num_rewards = token_level_rewards.shape
    
    # 2. Group-wise normalization for each reward component separately
    # GDPO: Decouple normalization per reward type before aggregation
    normalized_advantages = torch.zeros_like(token_level_rewards)
    
    # Organize indices by prompt ID (index)
    id2indices = defaultdict(list)
    # for i in range(bsz):
    #     # Handle tensor keys safely
    #     idx_key = index[i].item() if index[i].numel() == 1 else index[i]
    #     id2indices[idx_key].append(i)
    
    for i in range(bsz):
        id2indices[index[i]].append(i)


    for k in range(num_rewards):
        reward_k = token_level_rewards[:, k]
        
        for idx, indices in id2indices.items():
            if len(indices) > 1:
                group_rewards = reward_k[indices]
                mean = group_rewards.mean()
                std = group_rewards.std()
                # Eq 4: A^(k)_ij = (r^(k)_ij - mean) / std
                normalized_advantages[indices, k] = (group_rewards - mean) / (std + eps)
            else:
                # If group size is 1, advantage is undefined. Set to 0.
                normalized_advantages[indices, k] = 0.0

    # 3. Sum normalized advantages
    # Eq 5: A^sum_ij = sum(A^(k)_ij)
    sum_advantages = normalized_advantages.sum(dim=-1) # (bs,)
    
    # 4. Batch-wise normalization of the summed advantages
    # Eq 6: A^_sum_ij = (A^sum_ij - batch_mean) / (batch_std + eps)
    batch_mean = sum_advantages.mean()
    batch_std = sum_advantages.std()
    
    final_advantages = (sum_advantages - batch_mean) / (batch_std + eps)
    
    # Broadcast to sequence length
    returns = final_advantages.unsqueeze(-1) * response_mask
    # shape: (bs, response_length)
    
    return returns, returns







# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns



@torch.no_grad()
def compute_crpo_outcome_advantage(
    token_level_rewards: torch.Tensor, # Shape: (B, Len, 2)
    response_mask: torch.Tensor, 
    index: torch.Tensor, 
    beta_task: float = 0.5, # Task weight
    beta_style: float = 0.5, # Style weight
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CRPO Advantage Calculation.
    Input token_level_rewards contains [Task_Reward, Scaled_Style_Reward] inside the last dimension.
    """
    # 1. Unpack rewards
    # Already placed in the last token in fit, so directly sum to extract (B, 2)
    scores_dual = token_level_rewards.sum(dim=1) 
    
    task_scores = scores_dual[:, 0]  # (B,) - Raw Task Scores
    style_scores = scores_dual[:, 1] # (B,) - Already Global Scaled Style Scores

    # 2. Calculate Task Advantage (GRPO logic: Group Normalize)
    # Group by Prompt ID (index)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = task_scores.shape[0]

    for i in range(bsz):
        id2score[index[i]].append(task_scores[i])

    for idx in id2score:
        # No need to assert > 1 here, because if not > 1, it just degenerates to 0, which is more robust without error
        samples = torch.tensor(id2score[idx]).to(task_scores.device)
        id2mean[idx] = torch.mean(samples)
        id2std[idx] = torch.std(samples)

    task_advantages = torch.zeros_like(task_scores)
    for i in range(bsz):
        # Task Advantage: Intra-group relative advantage
        if len(id2score[index[i]]) > 1:
             task_advantages[i] = (task_scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
        else:
             task_advantages[i] = 0.0 # Or keep the original value, depending on the strategy



    # 2. Calculate Style Advantage (GRPO logic: Group Normalize)
    # Group by Prompt ID (index)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = style_scores.shape[0]

    for i in range(bsz):
        id2score[index[i]].append(style_scores[i])

    for idx in id2score:
        # No need to assert > 1 here, because if not > 1, it just degenerates to 0, which is more robust without error
        samples = torch.tensor(id2score[idx]).to(style_scores.device)
        id2mean[idx] = torch.mean(samples)
        id2std[idx] = torch.std(samples)

    style_advantages = torch.zeros_like(style_scores)
    for i in range(bsz):
        # Style Advantage: Intra-group relative advantage again
        if len(id2score[index[i]]) > 1:
             style_advantages[i] = (style_scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
        else:
             style_advantages[i] = 0.0 # Or keep the original value, depending on the strategy

    # # 3. Calculate Style Advantage (Global Anchor logic)
    # # Style Advantage uses the Global Scaled Score calculated in fit directly
    # # If style_scores_scaled > 0, it means it is more like the role than usual; < 0 means role drift occurred
    # style_advantages = style_scores_scaled

    # 4. Fuse advantages
    # Adv = \beta_1 * A_Task + \beta_2 * A_Style
    final_advantages = beta_task * task_advantages + beta_style * style_advantages

    # 5. Broadcast back to Token Level (only where Mask is valid)
    # Note: Returns are usually used as Target for Value Network in PPO.
    # Here, if CRPO does not train Critic, returns are actually only used for Logging.
    returns = final_advantages.unsqueeze(-1) * response_mask
    advantages = final_advantages.unsqueeze(-1) * response_mask
    
    return advantages, returns





@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float | torch.Tensor,
    clip_ratio_dual: float,
    adv_estimator: str,
    scale_factor_tensor: torch.Tensor,
    identity_log_probs_seq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the policy loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy

    """


    negative_approx_kl = log_probs - old_log_probs



    ratio_mean = None
    ratio_seq_mean = None
    scaling_factor = 1


    probs_role=None
    entropy_identity=None
    entropy_identity_weight=None
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    if identity_log_probs_seq is not None:
        log_probs_seq = torch.sum(log_probs * response_mask, dim=-1) / seq_lengths
        probs_role = torch.exp(log_probs_seq)/(torch.exp(identity_log_probs_seq)+torch.exp(log_probs_seq))
        entropy_identity=-(probs_role*torch.log(probs_role)+(1-probs_role)*torch.log((1-probs_role)))
        entropy_identity_weight=(1-0.02*entropy_identity)


    if adv_estimator=="gspo" or adv_estimator == "crpo":
        # seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
        negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
        log_seq_importance_ratio = log_probs - log_probs.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
        log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  
        ratio = torch.exp(log_seq_importance_ratio)
        clipped_ratio=torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)

        # print("negative_approx_kl_seq:",negative_approx_kl_seq.shape)
        # If it is a 1D tensor, need to unsqueeze to (bs, 1)
        if entropy_identity_weight is not None and entropy_identity_weight.dim() == 1:
            entropy_identity_weight = entropy_identity_weight.unsqueeze(-1)


        if adv_estimator == "crpo" and entropy_identity_weight is not None:    
            entropy_identity_weight=entropy_identity_weight.detach()
            advantages=advantages*entropy_identity_weight


        # clamp the ratio before exp to avoid nan
        # see: https://github.com/pytorch/pytorch/issues/10729
    else:
        ratio = torch.exp(negative_approx_kl)

        device = negative_approx_kl.device
        if isinstance(clip_ratio_high, float):
            clip_ratio_high = torch.tensor(clip_ratio_high, device=device)
        max_val = torch.log(1.0 + clip_ratio_high).to(device)

        min_val_scalar = float(np.log(1.0 - clip_ratio_low))
        min_val = torch.tensor(
            min_val_scalar, 
            device=device, 
            dtype=max_val.dtype
        )

        # print(negative_approx_kl.device,min_val.device,max_val.device)

        clipped_ratio = torch.exp(
            torch.clamp(negative_approx_kl, min_val, max_val)
        )

    ratio_mean=VF.masked_mean(ratio, response_mask).detach()

    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio 
    pg_loss3 = -advantages * clip_ratio_dual

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
    pg_clipfrac_higher = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    pg_clipfrac_lower = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    if adv_estimator=="crpo" or adv_estimator=="dapo":
        final_pg_loss = VF.masked_mean(final_pg_loss, response_mask)
    elif adv_estimator=="drgrpo":
        final_pg_loss = (final_pg_loss * response_mask).sum(dim=1).mean()
    else:
        final_pg_loss = VF.masked_mean(final_pg_loss, response_mask, dim=1).mean()



    pg_clipfrac_higher = VF.masked_mean(pg_clipfrac_higher, response_mask)
    pg_clipfrac_lower = VF.masked_mean(pg_clipfrac_lower, response_mask)
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)
    
    return final_pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl, ratio_mean, probs_role, entropy_identity


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    cliprange_value: float,
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    vf_loss = 0.5 * VF.masked_mean(torch.max(vf_loss1, vf_loss2), action_mask)  # clip if vf_loss1 < vf_loss2
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), action_mask)
    return vf_loss, vf_clipfrac


def compute_kl(log_probs: torch.FloatTensor, ref_log_probs: torch.FloatTensor, kl_penalty: str) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        kl = ref_log_probs - log_probs
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
