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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
# from ...utils.attention_utils import index_first_axis, pad_input, unpad_input
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]



class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits
        
        self.kl_coef_tensor=None
        self.scale_factor_tensor=None

        self.adv_estimator=None

        

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=2)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs


    def update_actor_advantage(self, data: DataProto):
        print(f"[Rank {self.rank}] Starting OAR Pre-computation...")
        
        input_ids_all = data.batch["input_ids"]
        advantages_all = data.batch["advantages"]
        
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "advantages"]
        non_tensor_select_keys = []
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]

        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        
        oar_weights_list = []
        

        if self.rank == 0:
            mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

        for mini_batch in mini_batches:
            micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
            
            if self.rank == 0:
                micro_batches = tqdm(micro_batches, desc="update_actor_advantage", position=3)

            for micro_batch in micro_batches:
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                curr_responses = model_inputs["responses"]
                curr_input_ids = model_inputs["input_ids"]
                curr_attn_mask = model_inputs["attention_mask"]
                curr_pos_ids = model_inputs["position_ids"]
            
                bsz, seqlen = curr_input_ids.shape
                res_len = curr_responses.size(1)
                start_r = seqlen - res_len
                
                # =======================================================
                # preprocess Padding Free 
                # =======================================================
                use_padding_free = self.config.padding_free
                
                if use_padding_free:
                    # 1. unpad Input IDs: [B, S] -> [Total_NNZ]
                    input_ids_rmpad, indices, *_ = unpad_input(
                        curr_input_ids.unsqueeze(-1), curr_attn_mask
                    )
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1) # [1, Total_NNZ]

                    # 2. process Position IDs
                    if curr_pos_ids.dim() == 3:
                        position_ids_rmpad = (
                            index_first_axis(rearrange(curr_pos_ids, "c b s ... -> (b s) c ..."), indices)
                            .transpose(0, 1).unsqueeze(1)
                        )
                    else:
                        position_ids_rmpad = index_first_axis(
                            rearrange(curr_pos_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                        ).transpose(0, 1)
                
                # =======================================================
                # 1. Teacher Forward (Eval Mode)
                # =======================================================
                self.actor_module.eval() 
                with torch.no_grad():
                    if use_padding_free:
                        out0 = self.actor_module(
                            input_ids=input_ids_rmpad,
                            attention_mask=None, # Padding free 
                            position_ids=position_ids_rmpad,
                            use_cache=False,
                            output_hidden_states=True, return_dict=True
                        )
                        # hidden_states
                        flat_hidden0 = out0.hidden_states[-1].squeeze(0) 
                        
                        last_hidden0 = pad_input(flat_hidden0.unsqueeze(-1), indices, bsz, seqlen).squeeze(-1)
                    else:
                        out0 = self.actor_module(
                            input_ids=curr_input_ids,
                            attention_mask=curr_attn_mask,
                            position_ids=curr_pos_ids,
                            use_cache=False,
                            output_hidden_states=True, return_dict=True
                        )
                        last_hidden0 = out0.hidden_states[-1]

                    # [Slice & Project] 
                    slice_hidden0 = last_hidden0[:, start_r-1 : seqlen-1, :]
                    slice_logits0 = self.actor_module.lm_head(slice_hidden0)
                    
                    p0_mean_logits = slice_logits0.mean(dim=0).float()
                    p0 = F.softmax(p0_mean_logits, dim=-1).detach()
                    
                    del out0, last_hidden0, slice_hidden0, slice_logits0
                    if use_padding_free: del flat_hidden0

                # =======================================================
                # 2. Embedding Perturbation
                # =======================================================
                emb_layer = self.actor_module.get_input_embeddings()
                inputs_embeds = emb_layer(curr_input_ids).detach()
                inputs_embeds.requires_grad_(True)
                
                noise = torch.zeros_like(inputs_embeds)
                noise[:, :start_r, :].normal_(mean=0.0, std=1e-3)
                noisy_embeds = inputs_embeds + noise
                
                # =======================================================
                # 3. Student Forward (Train Mode for GC)
                # =======================================================
                self.actor_module.train() 
                
                with torch.enable_grad():
                    if use_padding_free:
                        
                        # Flatten: [B, S, H] -> [B*S, H]
                        flat_embeds = rearrange(noisy_embeds, "b s h -> (b s) h")
                        # Select Non-Padding: [Total_NNZ, H]
                        inputs_embeds_rmpad = flat_embeds[indices]
                        # Reshape for model: [1, Total_NNZ, H]
                        inputs_embeds_rmpad = inputs_embeds_rmpad.unsqueeze(0)

                        out = self.actor_module(
                            inputs_embeds=inputs_embeds_rmpad,
                            attention_mask=None,
                            position_ids=position_ids_rmpad,
                            use_cache=False,
                            output_hidden_states=True, return_dict=True
                        )
                        flat_hidden = out.hidden_states[-1].squeeze(0)
                        
                        
                        last_hidden = pad_input(flat_hidden.unsqueeze(-1), indices, bsz, seqlen).squeeze(-1)
                    else:
                        out = self.actor_module(
                            inputs_embeds=noisy_embeds,
                            attention_mask=curr_attn_mask,
                            position_ids=curr_pos_ids,
                            use_cache=False,
                            output_hidden_states=True, return_dict=True
                        )
                        last_hidden = out.hidden_states[-1]
                    
                    
                    student_slice_hidden = last_hidden[:, start_r-1 : seqlen-1, :]
                    student_slice_logits = self.actor_module.lm_head(student_slice_hidden)
                    
                    p_mean_logits = student_slice_logits.mean(dim=0)
                    logp = F.log_softmax(p_mean_logits.float(), dim=-1)
                    
                    J = torch.sum(p0 * (torch.log(p0 + 1e-12) - logp))
                    
                    
                    grads = torch.autograd.grad(J, inputs_embeds, retain_graph=False)[0]
                    
                    del out, last_hidden, student_slice_hidden, student_slice_logits, J, p0, logp
                    if use_padding_free: del inputs_embeds_rmpad, flat_embeds, flat_hidden
                        
                # =======================================================
                # 4. Saliency Calculation 
                # =======================================================
                response_grads = grads[:, start_r:seqlen, :]
                response_embeds = inputs_embeds[:, start_r:seqlen, :].detach()
                sal = (response_grads * response_embeds).sum(dim=-1).abs().detach().float()
                
                # Normalization
                for b in range(bsz):
                    single_sal = sal[b]
                    eps = 1e-8
                    log_score = torch.log1p(single_sal)
                    min_v = log_score.min()
                    max_v = log_score.max()
                    range_v = (max_v - min_v) if max_v > min_v else 1.0
                    norm_imp = (log_score - min_v) / (range_v + eps)
                    
                    q_val = torch.quantile(norm_imp, 0.7)
                    w = torch.ones_like(norm_imp)
                    
                    mask_l = norm_imp < q_val
                    w[mask_l] = torch.clamp(norm_imp[mask_l] / (q_val + eps), min=0.1)
                    mask_h = ~mask_l
                    if mask_h.any():
                        w[mask_h] = 1.0 + 1.0 * (norm_imp[mask_h] - q_val) / (1.0 - q_val + eps)
                    
                    denom = w.sum()
                    if denom > 0:
                        w = w * (w.size(0) / (denom + eps))
                    
                    oar_weights_list.append(w)

                del grads, inputs_embeds, noisy_embeds, sal
                if use_padding_free: del input_ids_rmpad, position_ids_rmpad, indices

        
        self.actor_module.train()
        
        
        oar_weights_tensor = torch.stack(oar_weights_list, dim=0).to(advantages_all.device)
        
        data.batch.unlock_()
        new_advantages = advantages_all * oar_weights_tensor
        
        del oar_weights_tensor, input_ids_all, advantages_all

        
        # return data.batch["advantages"]
        
        return new_advantages.detach()





    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()


        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        
        if self.adv_estimator=="crpo":
            if "scale_factor_tensor" in data.batch.keys():
                select_keys.append("scale_factor_tensor")
                select_keys.append("kl_coef_tensor")
        if "responses_identity" in data.batch.keys():
            select_keys.append("responses_identity")
            select_keys.append("attention_mask_identity")
            select_keys.append("position_ids_identity")
            select_keys.append("response_mask_identity")
            select_keys.append("input_ids_identity")

        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=3)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]




                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

                    # 角色识别熵
                    identity_log_probs_seq=None
                    if "responses_identity" in model_inputs.keys():
                        with torch.no_grad():
                            self.actor_module.eval()
                            model_inputs_identity = dict()
                            model_inputs_identity["input_ids"] = model_inputs["input_ids_identity"]
                            model_inputs_identity["responses"] = model_inputs["responses_identity"]
                            model_inputs_identity["attention_mask"] = model_inputs["attention_mask_identity"]
                            model_inputs_identity["position_ids"] = model_inputs["position_ids_identity"]
                            model_inputs_identity["response_mask"] = model_inputs["response_mask_identity"]
                            identity_log_probs = self._forward_micro_batch(model_inputs_identity, temperature=temperature)
                            seq_lengths = torch.sum(model_inputs_identity["response_mask"], dim=-1).clamp(min=1)
                            identity_log_probs_seq = torch.sum(identity_log_probs * model_inputs_identity["response_mask"], dim=-1) / seq_lengths
                            identity_log_probs_seq = identity_log_probs_seq.detach()
                            self.actor_module.train()
                        del model_inputs_identity, identity_log_probs


                    entropy_loss = -VF.masked_mean(log_probs, response_mask)  # estimator of entropy loss


                    if self.adv_estimator=="crpo" or self.adv_estimator=="dapo":
                        clip_ratio_high=self.config.clip_ratio_high
                    else:
                        clip_ratio_high=self.config.clip_ratio_low

                    scale_factor_tensor=None
                    if "scale_factor_tensor" in model_inputs.keys():
                        # print("model_inputs[scale_factor_tensor].shape",model_inputs["scale_factor_tensor"].shape)
                        clip_ratio_high=self.config.clip_ratio_high
                        # metrics["actor/clip_ratio_high"] = clip_ratio_high.detach().mean().item()
                        metrics["actor/clip_ratio_high"] = clip_ratio_high
                        scale_factor_tensor=model_inputs["scale_factor_tensor"]
                    else:
                        clip_ratio_high=self.config.clip_ratio_high
                        metrics["actor/clip_ratio_high"] = clip_ratio_high


                    pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl, ratio_mean, probs_role, entropy_identity = core_algos.compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        adv_estimator=self.adv_estimator,
                        scale_factor_tensor=scale_factor_tensor,
                        identity_log_probs_seq=identity_log_probs_seq
                    )
                    if "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = core_algos.compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        # self.kl_coef_tensor=self.kl_coef_tensor.to(kld.device)
                        # print(kld.device,self.kl_coef_tensor.device)

                        # print("kld.shape",kld.shape)
                        kld_ori = VF.masked_mean(kld, response_mask)

                        metrics["actor/kl_ori"] = kld_ori.detach().item()

                        kld = kld * model_inputs["kl_coef_tensor"] if "kl_coef_tensor" in model_inputs.keys() else kld * self.config.kl_coef

                        # print("kld.shape",kld.shape)

                        kl_loss = VF.masked_mean(kld, response_mask)
                        if self.adv_estimator!="dapo":
                            pg_loss = pg_loss + kl_loss

                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        # if "kl_coef_tensor" in model_inputs.keys():
                        #     metrics["actor/kl_coef_mean"] = model_inputs["kl_coef_tensor"].mean()
                        if self.adv_estimator!="crpo":
                            metrics["actor/kl_coef"] = self.config.kl_coef
                        metrics["actor/ratio_mean"] = ratio_mean.mean().detach().cpu().item()
                        if probs_role is not None:
                            metrics["actor/probs_role"] = probs_role.mean().detach().cpu().item()
                        if entropy_identity is not None:
                            metrics["actor/entropy_identity"] = entropy_identity.mean().detach().cpu().item()

                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
