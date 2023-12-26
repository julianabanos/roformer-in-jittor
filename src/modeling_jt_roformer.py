# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Jittor RoFormer model. """

import math
import os
from typing import Optional

import numpy as np

# import torch
# import torch.utils.checkpoint
# from torch import nn
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import jittor as jt
from jittor import nn
from jittor.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    ModelOutput
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    SequenceSummary,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.utils import logging
from .configuration_roformer import RoFormerConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "junnyu/roformer_chinese_base"
_CONFIG_FOR_DOC = "RoFormerConfig"
_TOKENIZER_FOR_DOC = "RoFormerTokenizer"

ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "junnyu/roformer_chinese_small",
    "junnyu/roformer_chinese_base",
    "junnyu/roformer_chinese_char_small",
    "junnyu/roformer_chinese_char_base",
    "junnyu/roformer_chinese_sim_char_small",
    "junnyu/roformer_chinese_sim_char_base",
    "junnyu/roformer_chinese_sim_char_ft_small",
    "junnyu/roformer_chinese_sim_char_ft_base",
    "junnyu/roformer_small_discriminator",
    "junnyu/roformer_small_generator",
    "junnyu/roformer_base_wwm_cluecorpussmall",
    "junnyu/roformer_v2_chinese_char_small",
    "junnyu/roformer_v2_chinese_char_base",
    "junnyu/roformer_v2_chinese_char_large",
    # See all RoFormer models at https://huggingface.co/models?filter=roformer
]

class Norm(nn.Module):
    def __init__(self, eps = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = jt.mean(jt.square(x), dim=-1, keepdims=True)
        return x / jt.sqrt(variance + self.eps)

def initializer(tensor, num_hidden_layers=12, order=2, gain=1.0):
    """使用截断正态分布初始化
    """
    shape = tensor.shape
    if shape[0] > 10000 or shape[0] < 10:
        hidden_size = shape[1]
    else:
        hidden_size = shape[0]
    gain *= num_hidden_layers**(-1. / order)
    std = 1.13684723 / hidden_size**0.5 * gain
    return nn.init.trunc_normal_(tensor, std=std)

@dataclass
class CausalLMOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None