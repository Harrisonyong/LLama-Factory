#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   EPencoder.py
@Time    :   2025/01/13 16:41:01
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

import torch
import torch.nn.functional as F
from .config import ExpertBasedPromptConfig

class DynamicExpert(torch.nn.Module):
    def __init__(self, virtual_token_dim, intermediate_size):
        super().__init__()
        self.hidden_size = virtual_token_dim
        self.intermediate_size = intermediate_size

        self.up_proj = torch.nn.Linear(virtual_token_dim, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, virtual_token_dim, bias=False)
        self.gate_proj = torch.nn.Linear(virtual_token_dim, intermediate_size, bias=False)

        self.act_fn = torch.nn.SiLU()
    # [num_virtual_token, virtual_token_dim]
    def forward(self, input_ids):
        gate = self.act_fn(self.gate_proj(input_ids))
        up = self.up_proj(input_ids)
        return self.down_proj(gate * up)

class StaticExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    # [num_virtual_token, virtual_token_dim]
    def forward(self, input_ids):
        return input_ids

class Router(torch.nn.Module):
    def __init__(self, virtual_token_dim: int, num_experts: int, ToP_K: int):
        super().__init__()
        self.num_experts = num_experts
        self.ToP_K = ToP_K
        self.gate_layer = torch.nn.Linear(
                virtual_token_dim, num_experts, bias=False).float()

    def forward(self, inputs: torch.Tensor):
        # (batch_size * num_vritual_token, virtual_token_dim)
        inputs = inputs.float()
        # (batch_size * num_vritual_token, n_experts)
        router_logits = self.gate_layer(inputs)
        # (batch_size * num_vritual_token, n_experts)
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        selected_router_weight, selected_expert_idx = torch.topk(
            router_weights, k=self.ToP_K, dim=1
        )
        selected_router_weight /= selected_router_weight.sum(dim=-1, keepdim=True)
        # expert_info: 记录选择专家的token和权重
        expert_info = dict()
        for expert_id in range(self.num_experts):
            token_ids, weight_ids = torch.where(selected_expert_idx == expert_id)
            expert_info[expert_id] = [
                token_ids,
                selected_router_weight[token_ids, weight_ids],
            ]
        return expert_info

class EBencoder(torch.nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for EP-tuning.

    Args:
        config ([`ExpertBasedPromptConfig`]): The configuration of the Expert Based prompt encoder.
    """
    def __init__(self, config: ExpertBasedPromptConfig):
        super().__init__()
        num_layers = config.num_layers
        self.virtual_token_dim = config.token_dim
        self.encoder_intermediate_size = config.encoder_hidden_size
        self.total_virtual_tokens = config.num_virtual_tokens
         
        self.num_experts = config.num_experts
        self.num_static_experts = config.num_static_experts
        self.top_k = config.top_k
        
        # embedding layer
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.virtual_token_dim)
        # expert layer
        expert = DynamicExpert(self.virtual_token_dim, self.encoder_intermediate_size)
        self.experts = torch.nn.ModuleList(
            [StaticExpert() for _ in range(self.num_static_experts)]
            + [expert for _ in range(self.num_experts - self.num_static_experts)]
        )
        self.router = Router(self.virtual_token_dim, self.num_experts, self.top_k)
        self.transform = torch.nn.Linear(self.virtual_token_dim, self.virtual_token_dim * 2 * num_layers)

    def forward(self, indices):
        # [batch_size, num_virtual_tokens]
        hidden_states = self.embedding(indices)
        
        batch_size, num_virtual_token, virtual_token_dim = hidden_states.shape
        # [batch_size, num_virtual_tokens, virtual_token_dim]
        hidden_states = hidden_states.view(-1, virtual_token_dim)
        # router
        #[batch_size * num_virtual_tokens, virtual_token_dim]
        expert_info = self.router(hidden_states)
        output = torch.zeros(
            (batch_size * num_virtual_token, virtual_token_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # expert
        for expert, token_ids_and_weight in expert_info.items():
            token_ids, weight = token_ids_and_weight
            weight = weight.unsqueeze(-1)
            tokens = hidden_states.index_select(0, token_ids)
            expert_output = self.experts[expert](tokens)
            expert_output *= weight
            output.index_add_(0, token_ids, expert_output)
        output = output.reshape(batch_size, num_virtual_token, virtual_token_dim)
        output = self.transform(output)
        return output
        
        