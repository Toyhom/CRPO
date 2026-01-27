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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from tensordict import TensorDict 
from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, reduce_metrics_remove_min_ratio


# Logic/Task rewards (using group normalization, GRPO)
TASK_REWARD_KEYS = ["focus", "focus_attr", "format"] 
# Style/Human-like rewards (using global normalization, Anchor)
STYLE_REWARD_KEYS = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge1", "rouge2", "rougeL"]


class RewardScaler:
    """
    Normalize rewards by maintaining running mean and standard deviation.
    """
    def __init__(self, shape, gamma=0.99, epsilon=1e-8):
        self.shape = shape  # shape should be (num_rewards,)
        self.gamma = gamma
        self.epsilon = epsilon
        # Use torch.Tensor and keep it on CPU
        self.running_mean = torch.zeros(shape, dtype=torch.float64)
        self.running_var = torch.ones(shape, dtype=torch.float64)

    def update(self, rewards: torch.Tensor):
        """
        Update running mean and variance with a new batch of rewards.
        rewards: A Tensor with shape (batch_size, num_rewards).
        """
        # Ensure computation on CPU and float64 for precision
        rewards = rewards.to(self.running_mean.device, dtype=torch.float64)
        
        batch_mean = torch.mean(rewards, dim=0)
        batch_var = torch.var(rewards, dim=0, unbiased=False)

        # Update using Exponential Moving Average
        self.running_mean = self.gamma * self.running_mean + (1 - self.gamma) * batch_mean
        self.running_var = self.gamma * self.running_var + (1 - self.gamma) * batch_var

    def scale(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Scale rewards using current running mean and variance.
        rewards: A Tensor with shape (batch_size, num_rewards).
        """
        original_device = rewards.device
        # Ensure computation is performed on CPU
        rewards = rewards.to(self.running_mean.device)
        
        scaled_rewards = (rewards - self.running_mean.to(rewards.device)) / torch.sqrt(self.running_var.to(rewards.device) + self.epsilon)
        return scaled_rewards.to(original_device)

    def get_stats(self):
        """
        Return current running mean and standard deviation.
        """
        return self.running_mean, torch.sqrt(self.running_var + self.epsilon)





class CharacterGroupedRewardScaler:
    """
    A manager that dynamically creates and maintains a separate RewardScaler for each role name (string).
    Natively supports passing role names as np.ndarray(dtype=object).
    """
    def __init__(self, shape, gamma=0.999, epsilon=1e-8, reward_names=None):
        self.scaler_shape = shape
        self.scaler_gamma = gamma
        self.scaler_epsilon = epsilon
        self.scalers: Dict[str, RewardScaler] = {}
        self.reward_names = reward_names


    def update_and_scale(self, rewards: torch.Tensor, role_names: np.ndarray) -> torch.Tensor:
        """
        A core method that groups, updates, and normalizes rewards based on role names.
        
        Args:
            rewards (torch.Tensor): Raw reward tensor with shape (batch_size, num_rewards).
            role_names (np.ndarray): NumPy array with shape (batch_size,), dtype=object,
                                     where each element is the string name of the role.

        Returns:
            torch.Tensor: Conditionally normalized reward tensor with shape (batch_size, num_rewards).
        """
        # Using .shape[0] aligns better with NumPy conventions
        if role_names.shape[0] != rewards.shape[0]:
            raise ValueError(f"Batch size of rewards ({rewards.shape[0]}) does not match length of role_names ({role_names.shape[0]}).")

        scaled_rewards = torch.empty_like(rewards)
        
        # set() can directly work on NumPy arrays to efficiently get unique values
        unique_roles_in_batch = set(role_names)

        for role_name in unique_roles_in_batch:
            if role_name not in self.scalers:
                print(f"First time encountering role: '{role_name}', creating new RewardScaler for it.")
                new_scaler = RewardScaler(
                    shape=self.scaler_shape,
                    gamma=self.scaler_gamma,
                    epsilon=self.scaler_epsilon
                )
                self.scalers[role_name] = new_scaler
            
            scaler = self.scalers[role_name]

            mask = (role_names == role_name)
            
            # Convert NumPy boolean mask to PyTorch tensor
            mask_tensor = torch.from_numpy(mask).to(rewards.device)

            rewards_for_role = rewards[mask_tensor]

            scaler.update(rewards_for_role)
            scaled_rewards_for_role = scaler.scale(rewards_for_role)

            scaled_rewards[mask_tensor] = scaled_rewards_for_role.to(torch.float32)
            
        return scaled_rewards

    def get_all_stats(self) -> Dict[str, torch.Tensor]:
        all_stats = {}
        for role_name, scaler in self.scalers.items():
            mean, std = scaler.get_stats()
            # Random, with 0.05 probability
            if len(self.scalers) > 20:
                gate = 0.05
            else:
                gate = 1
            if np.random.rand() < gate:
                for i, reward_name in enumerate(self.reward_names):
                    all_stats[f'stats/mean/{role_name}/{reward_name}'] = mean[i].item()
                    all_stats[f'stats/std/{role_name}/{reward_name}'] = std[i].item()
        return all_stats



class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"
    CRPO = "crpo"

    DRGRPO = "drgrpo"
    DAPO = "dapo"
    GSPO = "gspo"
    GDPO = "gdpo"
    OAR = "oar"

    


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def compute_kl_coef(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]
    device = token_level_scores.device
    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    # Get role list
    role_names = data.non_tensor_batch["role_name"]
    

    # # We need to construct a (batch_size, 1) coefficient tensor because each sample has a different role and thus a different coefficient
    coef_list = [kl_ctrl.get_coef(r) for r in role_names]
    # kl_coef_tensor = torch.tensor(coef_list, device=device, dtype=token_level_scores.dtype).view(-1, 1)

    # scale_factor_list = [kl_ctrl.get_scale_factor(r) for r in role_names]
    # scale_factor_tensor = torch.tensor(scale_factor_list, device=device, dtype=token_level_scores.dtype).view(-1, 1)

    # Calculate approximate entropy: -mean(log_probs) over valid tokens
    ref_log_probs = data.batch["ref_log_probs"] 
    valid_token_cnt = response_mask.sum(dim=1)
    # (Batch_Size, )

    approx_entropy = -(ref_log_probs * response_mask).sum(dim=1) / (valid_token_cnt + 1e-8)
    
    # Calculate average KL for each sample (Batch_Size, )
    mean_kl_per_sample = kld.sum(dim=1) / (valid_token_cnt + 1e-8)
    
    # C. Update controller grouped by role
    unique_roles = np.unique(role_names)
    

    for role in unique_roles:
        # Find index mask for this role in the current Batch
        role_indices = np.where(role_names == role)[0]
        if len(role_indices) == 0:
            continue
            
        # Extract mean KL and Entropy for this role
        role_mean_kl = mean_kl_per_sample[role_indices].mean().item()
        role_mean_entropy = approx_entropy[role_indices].mean().item()
        
        # Update controller (Role-wise update)
        kl_ctrl.update(
            current_kl=role_mean_kl, 
            n_steps=len(role_indices), # Number of steps can be replaced by number of samples, affecting EMA smoothing
            role_name=role,
            current_entropy=role_mean_entropy
        )

    # coef_list = [kl_ctrl.get_coef(r) for r in role_names]
    kl_coef_tensor = torch.tensor(coef_list, device=device, dtype=token_level_scores.dtype).view(-1, 1)

    scale_factor_list = [kl_ctrl.get_scale_factor(r) for r in role_names]
    scale_factor_tensor = torch.tensor(scale_factor_list, device=device, dtype=token_level_scores.dtype).view(-1, 1)

    # Record some metrics for logging
    avg_coef = np.mean(coef_list)
    total_kl = mean_kl_per_sample.mean().item()
    total_entropy = approx_entropy.mean().item()
    
    global_avg_entropy = kl_ctrl.global_avg_entropy
    role_entropy_ma = kl_ctrl.role_entropy_ma

    metrics_role = {}

    for role in unique_roles:
        metrics_role[f"actor/{role}_entropy_ma"] = role_entropy_ma[role]


    # Logging
    metrics = {
        "actor/old_kl": total_kl, 
        "actor/kl_coef": avg_coef, # Record average coefficient
        "actor/entropy_mean": total_entropy,
        "actor/global_avg_entropy": global_avg_entropy
    }

    metrics.update(metrics_role)

    return scale_factor_tensor, kl_coef_tensor, metrics

def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]
    device = token_level_scores.device
    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    if isinstance(kl_ctrl, core_algos.EntropyAwareRoleKLController):
        # Get role list
        role_names = data.non_tensor_batch["role_name"]
        
        # A. Vectorized application of KL coefficients
        # We need to construct a (batch_size, 1) coefficient tensor because each sample has a different role and thus a different coefficient
        coef_list = [kl_ctrl.get_coef(r) for r in role_names]
        kl_coef_tensor = torch.tensor(coef_list, device=device, dtype=token_level_scores.dtype).view(-1, 1)
        
        # Apply penalty: rewards = scores - coef[i] * kld[i]
        data.batch["token_level_rewards"] = token_level_scores - kl_coef_tensor * kld
        
        # B. Prepare statistics required for updating controller
        # Calculate approximate entropy: -mean(log_probs) over valid tokens
        ref_log_probs = data.batch["ref_log_probs"] 
        valid_token_cnt = response_mask.sum(dim=1)
        # (Batch_Size, )
        approx_entropy = -(ref_log_probs * response_mask).sum(dim=1) / (valid_token_cnt + 1e-8)
        
        # Calculate average KL for each sample (Batch_Size, )
        mean_kl_per_sample = kld.sum(dim=1) / (valid_token_cnt + 1e-8)
        
        # C. Update controller grouped by role
        # Need to aggregate on CPU here
        unique_roles = np.unique(role_names)
        
        # Record some metrics for logging
        avg_coef = np.mean(coef_list)
        total_kl = mean_kl_per_sample.mean().item()
        total_entropy = approx_entropy.mean().item()
        
        for role in unique_roles:
            # Find index mask for this role in the current Batch
            role_indices = np.where(role_names == role)[0]
            if len(role_indices) == 0:
                continue
                
            # Extract mean KL and Entropy for this role
            role_mean_kl = mean_kl_per_sample[role_indices].mean().item()
            role_mean_entropy = approx_entropy[role_indices].mean().item()
            
            # Update controller (Role-wise update)
            kl_ctrl.update(
                current_kl=role_mean_kl, 
                n_steps=len(role_indices), # Number of steps can be replaced by number of samples, affecting EMA smoothing
                role_name=role,
                current_entropy=role_mean_entropy
            )
            
        # Logging
        metrics = {
            "critic/kl": total_kl, 
            "critic/kl_coef_mean": avg_coef, # Record average coefficient
            "critic/entropy_mean": total_entropy
        }

    else:
        data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

        current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
        current_kl = torch.mean(current_kl, dim=0).item()
        metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

        # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
        kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, config: PPOConfig, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.CRPO:
        if config.algorithm.wo_norm:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
        else:
            advantages, returns = core_algos.compute_crpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.DRGRPO:
        advantages, returns = core_algos.compute_drgrpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.DAPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.GSPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.GDPO:
        advantages, returns = core_algos.compute_gdpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.OAR:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError(f"AdvantageEstimator {adv_estimator} is not implemented.")

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")


        print(f"Total training steps: {self.training_steps}")

        self.reward_names = [
            "format", "bleu1", "bleu2", "bleu3",
            "bleu4", "rouge1", "rouge2", "rougeL", "focus", "focus_attr"
        ]

        # self.reward_names_style = [
        #     "bleu1", "bleu2", "bleu3",
        #     "bleu4", "rouge1", "rouge2", "rougeL", "format"
        # ]

        if self.config.algorithm.reward_scaler and self.config.algorithm.adv_estimator != "crpo":
            global STYLE_REWARD_KEYS
            STYLE_REWARD_KEYS = ["format", "bleu1", "bleu2", "bleu3", "bleu4", "rouge1", "rouge2", "rougeL", "focus", "focus_attr"]
        
        
        self.grouped_scaler = CharacterGroupedRewardScaler(shape=(len(STYLE_REWARD_KEYS),), gamma=0.99, reward_names=STYLE_REWARD_KEYS)

        self.reward_scaler = RewardScaler(shape=(len(STYLE_REWARD_KEYS),))




        self.reward_weights_task = torch.tensor([
            0.35, # focus
            0.2, # focus_attr
            0.1  # format
        ])

        self.reward_weights_style = torch.tensor([
            0.1, # bleu1
            0.05, # bleu2
            0.02, # bleu3
            0.01,  # bleu4
            0.1,  # rouge1
            0.05,  # rouge2
            0.02  # rougeL
        ])


        self.reward_weights = torch.tensor([
            0.1,  # format
            0.1, # bleu1
            0.05, # bleu2
            0.02, # bleu3
            0.01,  # bleu4
            0.1,  # rouge1
            0.05,  # rouge2
            0.02,  # rougeL
            0.35, # focus
            0.2, # focus_attr
        ])

        # Ensure sum of weights is 1
        # self.reward_weights = self.reward_weights / self.reward_weights.sum()




    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            reward_tensor = reward_tensor["overall"]

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        return {"val/reward_score": reward_score, **val_reward_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        # First get the original path
        base_path = self.config.trainer.load_checkpoint_path
        # Check if the original path is a directory and not a checkpoint folder itself
        # This way, if the user explicitly specifies '.../global_step_10000', this logic will be skipped, respecting the user's precise specification
        if os.path.isdir(base_path) and 'global_step_' not in os.path.basename(base_path):
            print(f"'{base_path}' is a directory, automatically searching for the latest checkpoint...")
            
            # 1. Find all subfolders in 'global_step_*' format
            try:
                subfolders = [
                    f for f in os.listdir(base_path) 
                    if f.startswith('global_step_') and os.path.isdir(os.path.join(base_path, f))
                ]
            except FileNotFoundError:
                raise FileNotFoundError(f"Specified checkpoint directory does not exist: {base_path}")
            if not subfolders:
                raise FileNotFoundError(f"No 'global_step_*' format checkpoint folders found in directory '{base_path}'.")
            # 2. Sort by step value (descending)
            # Use lambda function to extract the number after 'global_step_' and convert to integer to ensure correct sorting
            # For example, it correctly handles global_step_10000 > global_step_2000
            subfolders.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
            
            # 3. Get the latest folder and update the path in config
            latest_checkpoint_folder = subfolders[0]
            self.config.trainer.load_checkpoint_path = os.path.join(base_path, latest_checkpoint_folder)
            
            print(f"Automatically selected the latest checkpoint: {self.config.trainer.load_checkpoint_path}")

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        for _ in tqdm(range(self.config.trainer.total_epochs), desc="Epoch", position=0):
            for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):


                self.global_step += 1
                if self.global_step > self.training_steps:
                    break

                metrics, timing_raw = {}, {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if "multi_modal_data" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                with timer("step", timing_raw):

                    
                    # generate a batch
                    with timer("gen", timing_raw):  # wg: worker group
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)                   

                    if self.config.algorithm.adv_estimator == "remax":
                        with timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["temperature"] = 0
                            gen_baseline_batch.meta_info["n"] = 1
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
                            reward_baseline_tensor = reward_baseline_tensor["overall"]
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.batch.unlock_()
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    # if self.config.algorithm.adv_estimator == "crpo" and not self.config.algorithm.wo_sample:
                    with timer("gen_crpo", timing_raw):

                        gen_baseline_batch = deepcopy(gen_batch)

                        # system_prompt_split_token = self.tokenizer.token_to_id("Character")
                        system_prompt_split_token = self.tokenizer.convert_tokens_to_ids("Character")

                        # [bts, seq]
                        ori_input_ids = gen_baseline_batch.non_tensor_batch["raw_prompt_ids"]


                        for i in range(ori_input_ids.shape[0]):
                            # Truncate content after system_prompt_split_token
                            row = ori_input_ids[i]

                            if not isinstance(row, torch.Tensor):
                                row = torch.tensor(row)
                            # Find position sequence of split token
                            sep_idx = (row == system_prompt_split_token).nonzero(as_tuple=True)[0]

                            # Find separator position
                            idx = sep_idx[-1].item()
                            ori_input_ids[i] = row[idx + 1:] 
                            
                        gen_baseline_batch.non_tensor_batch["raw_prompt_ids"]=ori_input_ids

                        # Non-role-playing
                        # gen_baseline_batch.meta_info["temperature"] = 0
                        gen_baseline_batch.meta_info["n"] = self.config.algorithm.anchor_n 
                        gen_baseline_batch.meta_info["temperature"] = 1.0
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                        print(f"DEBUG: Output Size: {gen_baseline_output.batch['input_ids'].shape[0]}, Anchor N: {self.config.algorithm.anchor_n}")

                        if not self.config.algorithm.wo_rlkl:
                            if self.config.algorithm.anchor_n==1:
                                batch.batch["attention_mask_identity"]=gen_baseline_output.batch["attention_mask"]
                                batch.batch["position_ids_identity"]=gen_baseline_output.batch["position_ids"]
                                identity_responses=gen_baseline_output.batch["responses"]
                                identity_mask=gen_baseline_output.batch["response_mask"]
                                batch.batch["responses_identity"]=identity_responses
                                batch.batch["response_mask_identity"]=identity_mask
                                batch.batch["input_ids_identity"]=gen_baseline_output.batch["input_ids"]
                            # Take only one
                            else:
                                batch_size_ori=gen_baseline_output.batch["input_ids"].shape[0]//self.config.algorithm.anchor_n
                                batch.batch["input_ids_identity"]=gen_baseline_output.batch["input_ids"].view(batch_size_ori, self.config.algorithm.anchor_n, -1)[:, 0, :]
                                batch.batch["attention_mask_identity"]=gen_baseline_output.batch["attention_mask"].view(batch_size_ori, self.config.algorithm.anchor_n, -1)[:, 0, :]
                                batch.batch["position_ids_identity"]=gen_baseline_output.batch["position_ids"].view(batch_size_ori, self.config.algorithm.anchor_n, -1)[:, 0, :]
                                batch.batch["responses_identity"]=gen_baseline_output.batch["responses"].view(batch_size_ori, self.config.algorithm.anchor_n, -1)[:, 0, :]
                                batch.batch["response_mask_identity"]=gen_baseline_output.batch["response_mask"].view(batch_size_ori, self.config.algorithm.anchor_n, -1)[:, 0, :]





                        if self.config.algorithm.adv_estimator == "crpo" and not self.config.algorithm.wo_sample:
                            new_batch_data = {}
                            for key in gen_batch_output.batch.keys():
                                A = gen_batch_output.batch[key]   # [3584, 8096]
                                B = gen_baseline_output.batch[key] # [512, 8096]
                                dim = A.shape[-1]

                                print("B.shape",B.shape)
                                
                                # 512
                                batch_size_ori = B.shape[0] // self.config.algorithm.anchor_n
                                # 7
                                temp_n = A.shape[0] // batch_size_ori 
                                A_view = A.view(batch_size_ori, temp_n, dim)
                                B_view = B.view(batch_size_ori, self.config.algorithm.anchor_n, dim)
                                combined = torch.cat([A_view, B_view], dim=1)
                                # Do not assign back to gen_batch_output.batch[key] directly, store in temporary dictionary instead
                                new_batch_data[key] = combined.flatten(0, 1) # [4096, 8096]


                            new_tensor_dict = TensorDict(
                                new_batch_data, 
                                batch_size=[batch_size_ori*(temp_n+self.config.algorithm.anchor_n)], 
                                device=gen_batch_output.batch.device
                            )




                            gen_batch_output.batch = new_tensor_dict

                            batch = batch.repeat(repeat_times=self.config.worker.rollout.n + self.config.algorithm.anchor_n, interleave=True)
                            batch = batch.union(gen_batch_output)
                            batch.non_tensor_batch.pop("multi_modal_data", None)
                            
                            del gen_baseline_batch, gen_baseline_output, ori_input_ids
                        else:
                            batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                            batch = batch.union(gen_batch_output)
                            batch.non_tensor_batch.pop("multi_modal_data", None)   

                            del gen_baseline_batch, gen_baseline_output, ori_input_ids  
                    # else:
                    #     # repeat to align with repeated responses in rollout
                    #     batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    #     batch = batch.union(gen_batch_output)
                    #     batch.non_tensor_batch.pop("multi_modal_data", None)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # compute reward
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)

                        batch.batch.unlock_() 

                        batch = batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with timer("adv", timing_raw):
                        # get token level scores

                        if self.config.algorithm.adv_estimator == "crpo" and not self.config.algorithm.wo_norm:


                            reward_tensors_dict, reward_metrics = ray.get(reward_ref)

                            # 1. Separate Task and Style rewards
                            # Assuming user_reward_weights already covers keys
                            
                            print("reward_tensors_dict",reward_tensors_dict["focus"].shape)

                            # print("reward_weights[name]",self.reward_weights)

                            # Aggregate Task Rewards (Raw)
                            task_rewards_sum = torch.stack(
                                [reward_tensors_dict[name].sum(dim=-1) 
                                 for name in TASK_REWARD_KEYS if name in reward_tensors_dict],
                                dim=1
                            )

                            # Aggregate Style Rewards (Raw)
                            style_rewards_sum = torch.stack(
                                [reward_tensors_dict[name].sum(dim=-1)
                                 for name in STYLE_REWARD_KEYS if name in reward_tensors_dict],
                                dim=1
                            )
                            # 2. Perform global role normalization on Style Rewards (Global Anchor)

                            del reward_tensors_dict

                            current_batch_size = task_rewards_sum.shape[0]
                            if self.config.algorithm.reward_scaler_method == "role":
                                role_names = batch.non_tensor_batch["role_name"] # or role_group
                            elif self.config.algorithm.reward_scaler_method == "group":
                                batch.non_tensor_batch["role_name"] = batch.non_tensor_batch["role_group"]
                                role_names = batch.non_tensor_batch["role_group"]
                            else:
                                role_names = np.full(current_batch_size, "1", dtype=object)

                            
                            # Note: update_and_scale will update internal EMA mean
                            # Meaning: We compare current generated Style score with this role's historical Style score
                            scaled_style_rewards = self.grouped_scaler.update_and_scale(
                                style_rewards_sum, # scaler expects (B, Num_Rewards)
                                role_names
                            )


                            # 4. Merge normalized rewards into final per-sequence reward using weights
                            scaled_style_rewards = (scaled_style_rewards * self.reward_weights_style.to(scaled_style_rewards.device)).sum(dim=-1) # shape: (batch_size,)


                            task_rewards_sum = (task_rewards_sum * self.reward_weights_task.to(task_rewards_sum.device)).sum(dim=-1) # shape: (batch_size,)


                            # 3. Record statistics (optional)
                            all_stats = self.grouped_scaler.get_all_stats()
                            metrics.update(all_stats)
                            
                            # 4. Store in Batch
                            # We need to store these separately to pass to compute_advantage
                            # Still using token_level_scores field, but stored in Stack format
                            # Channel 0: Raw Task Reward (To be Group processed)
                            # Channel 1: Scaled Style Reward (Already Global processed)
                            
                            # Broadcast back to token level (only at the last token)
                            device = batch.batch["response_mask"].device
                            response_mask = batch.batch["response_mask"].to(device)
                            last_token_indices = (response_mask.sum(dim=1).long() - 1).clamp(min=0)
                            batch_indices = torch.arange(response_mask.size(0), device=device)
                            
                            # Create (Batch, Len, 2) tensor to store dual-stream rewards
                            dual_stream_rewards = torch.zeros(
                                (response_mask.size(0), response_mask.size(1), 2), 
                                device=device, dtype=torch.float32
                            )
                            
                            dual_stream_rewards[batch_indices, last_token_indices, 0] = task_rewards_sum.float()
                            dual_stream_rewards[batch_indices, last_token_indices, 1] = scaled_style_rewards.float()
                            
                            batch.batch["token_level_scores"] = dual_stream_rewards # Overwrite original field

                        # Character-R1
                        elif self.config.algorithm.reward_scaler and self.config.algorithm.adv_estimator != "crpo":

                            reward_tensors_dict, reward_metrics = ray.get(reward_ref)

                            print("reward_tensors_dict",reward_tensors_dict["focus"].shape)

                            # Aggregate
                            style_rewards_sum = torch.stack(
                                [reward_tensors_dict[name].sum(dim=-1)
                                 for name in STYLE_REWARD_KEYS if name in reward_tensors_dict],
                                dim=1
                            )

                            del reward_tensors_dict

                            current_batch_size = style_rewards_sum.shape[0]
                            if self.config.algorithm.reward_scaler_method == "role":
                                role_names = batch.non_tensor_batch["role_name"] # or role_group
                            elif self.config.algorithm.reward_scaler_method == "group":
                                batch.non_tensor_batch["role_name"] = batch.non_tensor_batch["role_group"]
                                role_names = batch.non_tensor_batch["role_group"]
                            else:
                                role_names = np.full(current_batch_size, "1", dtype=object)

                            
                            scaled_style_rewards = self.grouped_scaler.update_and_scale(
                                style_rewards_sum, # scaler (B, Num_Rewards)
                                role_names
                            )


                            scaled_style_rewards = (scaled_style_rewards * self.reward_weights.to(scaled_style_rewards.device)).sum(dim=-1) # shape: (batch_size,)

                            all_stats = self.grouped_scaler.get_all_stats()
                            metrics.update(all_stats)
                            
                            device = batch.batch["response_mask"].device
                            response_mask = batch.batch["response_mask"].to(device)
                            last_token_indices = (response_mask.sum(dim=1).long() - 1).clamp(min=0)
                            batch_indices = torch.arange(response_mask.size(0), device=device)
                            
                            dual_stream_rewards = torch.zeros(
                                (response_mask.size(0), response_mask.size(1)), 
                                device=device, dtype=torch.float32
                            )
                            
                            dual_stream_rewards[batch_indices, last_token_indices] = scaled_style_rewards.float()
                            
                            batch.batch["token_level_scores"] = dual_stream_rewards # Overwrite original field

                        elif self.config.algorithm.adv_estimator == "gdpo":
                            reward_tensors_dict, reward_metrics = ray.get(reward_ref)

                            token_level_scores = torch.stack(
                                [reward_tensors_dict[name].sum(dim=-1) for name in reward_tensors_dict],
                                dim=1
                            )
                            # shape: (batch_size, num_rewards)
                            
                            batch.batch["token_level_scores"] = token_level_scores # Overwrite original field


                        else:
                            if self.config.algorithm.adv_estimator == "crpo":
                                batch.non_tensor_batch["role_name"] = batch.non_tensor_batch["role_group"]
                                role_names = batch.non_tensor_batch["role_group"]

                            reward_tensor, reward_metrics = ray.get(reward_ref)

                            batch.batch["token_level_scores"] = reward_tensor["overall"]



                        if self.config.algorithm.adv_estimator == "crpo" and not self.config.algorithm.wo_sample:
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics_remove_min_ratio(reward_metrics,ratio=self.config.algorithm.anchor_n/(self.config.worker.rollout.n+self.config.algorithm.anchor_n)).items()}
                        else:
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.algorithm.adv_estimator == "crpo" and not self.config.algorithm.wo_rlkl:
                            scale_factor_tensor, kl_coef_tensor, kl_metrics = compute_kl_coef(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            batch.batch["kl_coef_tensor"] = kl_coef_tensor
                            batch.batch["scale_factor_tensor"] = scale_factor_tensor
                            # self.actor_rollout_wg.update_factor(kl_coef_tensor, scale_factor_tensor)
                            metrics.update(kl_metrics)

                        response_mask = batch.batch["response_mask"]

                        ref_log_probs = batch.batch["ref_log_probs"] 
                        valid_token_cnt = response_mask.sum(dim=1)
                        approx_entropy_ref = -(ref_log_probs * response_mask).sum(dim=1) / (valid_token_cnt + 1e-8)

                        old_log_probs = batch.batch["old_log_probs"] 
                        valid_token_cnt = response_mask.sum(dim=1)
                        approx_entropy_old = -(old_log_probs * response_mask).sum(dim=1) / (valid_token_cnt + 1e-8)

                        entropy_metrics = {
                            "actor/entropy_ref": approx_entropy_ref.mean().item(),
                            "actor/entropy_old": approx_entropy_old.mean().item()
                        }

                        metrics.update(entropy_metrics)
                            




                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            config=self.config,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            )
                    
                    self.actor_rollout_wg.update_adv_estimator(self.config.algorithm.adv_estimator)

                    # update critic
                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_step:
                        if self.config.algorithm.adv_estimator == "oar":
                            batch.batch.unlock_() 
                            # 1. Receive the returned "slimmed down" DataProto
                            output_data_proto = self.actor_rollout_wg.update_actor_advantage(batch)
                            # 2. Extract and overwrite
                            batch.batch["advantages"] = output_data_proto.batch["advantages"]

                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                num_gpus = self.resource_pool_manager.get_num_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

                self.logger.log(data=metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
