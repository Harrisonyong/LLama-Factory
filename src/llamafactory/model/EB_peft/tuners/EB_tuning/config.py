#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2025/01/13 14:26:27
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

from dataclasses import dataclass, field

from peft.config import PromptLearningConfig
from peft.utils import PeftType


    
@dataclass
class ExpertBasedPromptConfig(PromptLearningConfig):
    """
    This is the base configuration class to store the configuration of Expert-based prompt tuning.

    Args:
        encoder_intermediate_size: The intermediate size of the expert encoder.
        num_experts: total number of experts: dynamic expert + static expert.
        num_static_experts: Number of static experts.
        top_k: topK of exeprts to be choiced
    """

    # exeprt intermediate size
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The intermediate size of the expert encoder"},
    )
    # total number of experts: dynamic expert + static expert
    num_experts: int = field(
        default=16,
        metadata={"help": "Toal number of experts"},
    ) 
    # number of static experts
    num_static_experts: int = field(
        default=1,
        metadata={"help": "Number of static experts"},
    )
    # TopK of exeprts to be choiced
    top_k: int = field(
        default=4,
        metadata={"help": "TopK of exeprts to be choiced"},
    )
    
    def __post_init__(self):
        self.peft_type = "EB_TUNING"
