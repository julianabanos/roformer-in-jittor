import math
import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np
import json
import os
from typing import Optional, Tuple, Callable
import warnings
import inspect
from typing import List, Set, Tuple
from dataclasses import dataclass
from .jt_dataclasses import DataClassEncoder, DataClassRoformer, DataClassCausalLM, DataClassSequenceClassifier
import pdb

jt.flags.use_cuda = 1
# roformer model

class ClassificationHead(Module):
    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def execute(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class ModelSeqClassifier(Module):
    def __init__(self, config):
        super(ModelSeqClassifier, self).__init__()
        self.roformer = Roformer(config)
        self.classifier = ClassificationHead(config)

    def execute(self, 
                input_ids=None, 
                attention_mask=None, 
                token_type_ids=None, 
                head_mask=None,                
                inputs_embeds=None, 
                labels=None, # new
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return DataClassSequenceClassifier(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




class ModelCausalLM(Module):
    def __init__(self, config):
        super(ModelCausalLM, self).__init__()
        self.roformer = Roformer(config)
        self.cls = Cls(config)

    def generate(self, input_ids, token_type_ids=None, attention_mask=None, top_p=0.95, max_length=128, do_sample=True):    
        # Assuming that the Roformer model's output can be used directly for token generation.
        # This method generates one token at a time using top-p sampling.

        # Start with the provided input_ids

        generated = input_ids

        # Iterate until max_length is reached
        for _ in range(max_length):
            # # Get the model's output
            # outputs = self.roformer(input_ids=generated, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, return_dict=True)
            with jt.no_grad():
                outputs = self(input_ids=generated, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False)
            
            # Assume the last layer output is the logits (adjust according to your model's specifics)
            # logits = outputs[0][:, -1, :]

            logits = outputs.logits[:, -1, :] # TODO: CHECK 
            # pdb.set_trace()

            # Apply top-p sampling to the logits to get the next token
            filtered_logits = self.top_p_filtering(logits, top_p)
            # filtered_logits = logits
            if do_sample:
                probabilities = nn.softmax(filtered_logits, dim=-1)
                next_token = jt.multinomial(probabilities, num_samples=1)
                # pdb.set_trace()
            else:
                # Use the most likely next token if do_sample is False
                next_token = jt.argmax(filtered_logits, dim=-1)

            # Concatenate the new token to the generated sequence
            generated = jt.cat((generated, next_token), dim=1)
            attention_mask = jt.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
            token_type_ids = jt.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

            # pdb.set_trace()

            # Stop if the sequence is getting too long
            if generated.size(1) > max_length:
                break

        return generated

    def top_p_filtering(self, logits, top_p):
        # Sort the logits to identify the cutoff for top-p
        sorted_logits, sorted_indices = jt.sort(logits, descending=True)
        cumulative_probs = jt.cumsum(nn.softmax(sorted_logits, dim=-1), dim=-1)
        # pdb.set_trace()

        # Remove tokens with a cumulative probability above the threshold (top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter the sorted indices to the original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        # pdb.set_trace()
        # print(logits.nonzero())
        return logits


    def execute(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None, encoder_hidden_states=None, 
                encoder_attention_mask=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, labels=None, 
                use_cache=None, output_attentions=True, output_hidden_states=True,): #TODO: CHECK
        
        # return_dict = (
        #     return_dict if return_dict is not None else self.config.use_return_dict
        # )

        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )
        # print 
        # pdb.set_trace()

        sequence_output = outputs.last_hidden_state #TODO:CHECK
        # print("sequence_output", sequence_output.shape)
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        # if not return_dict:
        #     output = (prediction_scores,) + outputs[1:] # with pooler
        #     return ((lm_loss,) + output) if lm_loss is not None else output
        # pdb.set_trace()

        return DataClassCausalLM(
            loss=lm_loss,
            logits=prediction_scores,
            pooler_output=outputs.pooler_output, # with pooler_output
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


#####################################################

class Roformer(Module):
    def __init__(self, config):
        super(Roformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.config = config
        self.dtype = "float32"
        self.add_pooling_layer = config.add_pooling_layer

        if self.add_pooling_layer:
            self.pooler = RoFormerPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def execute(self, input_ids=None, attention_mask=None, token_type_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None,
                use_cache=None, output_attentions=True, output_hidden_states=True, return_dict=True):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = jt.ones(
                ((batch_size, seq_length + past_key_values_length))
            )
        if token_type_ids is None:
            token_type_ids = jt.zeros(input_shape, dtype=self.dtype)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: jt.Var = self.get_extended_attention_mask(
            attention_mask, input_shape, past_key_values_length
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = jt.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # if hasattr(self, "embeddings_project"):
        #     embedding_output = self.embeddings_project(embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.add_pooling_layer else None

        # pdb.set_trace()
        return DataClassRoformer(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        # return (sequence_output, pooled_output) + encoder_outputs[1:]
        

    # 添加了个past_key_values_length
    def get_extended_attention_mask(self, attention_mask: jt.Var, input_shape, past_key_values_length):
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            if self.config.is_decoder and past_key_values_length > 0: # 第一次编码的时候不需要使用decoder mask，之后的需要decoder mask。
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        batch_size, seq_length = input_shape
        seq_ids = jt.arange(seq_length)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = jt.cat(
                [
                    jt.ones((batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask
    
    def get_head_mask(
        self, head_mask: Optional[jt.Var], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> jt.Var:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
    
    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.ndim == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.ndim == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.ndim == 5, f"head_mask.dim != 5, instead {head_mask.ndim}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

# DONE
class Embeddings(Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # LayerNorm
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps) if config.norm_type == 'layer_norm' else Norm(eps=config.layer_norm_eps)
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def execute(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = jt.zeros(
                input_shape, dtype=jt.to(jt.int64) , device=inputs_embeds.device
            ) # torch.long is equivalent to to(torch.int64)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter): # TODO: change to just jt.Var?
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = jt.float32(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = jt.float32(np.cos(position_enc[:, 1::2]))
        out.detach_inplace()
        return out

    @jt.no_grad()
    def execute(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = jt.arange(
            start=past_key_values_length,
            end=past_key_values_length + seq_len,
            dtype=jt.int64,
        )
        return super().execute(positions)


class Encoder(Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
        )
        self.layer = nn.ModuleList(
            [Layer(config) for _ in range(12)]
        )
        self.gradient_checkpointing = False

    def execute(
            self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        sinusoidal_pos = self.embed_positions(hidden_states.shape[1], past_key_values_length)[
            None, None, :, :
        ].chunk(2, dim=-1)

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                sinusoidal_pos,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            # pdb.set_trace()

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
        # return tuple(
        #     v
        #     for v in [
        #         hidden_states,
        #         next_decoder_cache,
        #         all_hidden_states,
        #         all_self_attentions,
        #         all_cross_attentions,
        #     ]
        #     if v is not None
        # )
        return DataClassEncoder(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class Layer(Module):
    def __init__(self, config):
        super(Layer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def execute(self, hidden_states, attention_mask=None, sinusoidal_pos=None,
                head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        # attention_output = self.attention(hidden_states, attention_mask)
        # intermediate_output = self.intermediate(attention_output)
        # layer_output = self.output(intermediate_output, attention_output)
        # return layer_output
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
                    "layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                sinusoidal_pos,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = self.apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def apply_chunking_to_forward(self,
        forward_fn: Callable[..., jt.Var], chunk_size: int, chunk_dim: int, *input_tensors
    ) -> jt.Var:
        """
        This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
        `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

        If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
        applying `forward_fn` to `input_tensors`.

        Args:
            forward_fn (`Callable[..., torch.Tensor]`):
                The forward function of the model.
            chunk_size (`int`):
                The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
            chunk_dim (`int`):
                The dimension over which the `input_tensors` should be chunked.
            input_tensors (`Tuple[torch.Tensor]`):
                The input tensors of `forward_fn` which will be chunked

        Returns:
            `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.


        Examples:

        ```python
        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states


        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
        ```"""

        assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

        # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
        num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
        if num_args_in_forward_chunk_fn != len(input_tensors):
            raise ValueError(
                f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
                "tensors are given"
            )

        if chunk_size > 0:
            tensor_shape = input_tensors[0].shape[chunk_dim]
            for input_tensor in input_tensors:
                if input_tensor.shape[chunk_dim] != tensor_shape:
                    raise ValueError(
                        f"All input tenors have to be of the same shape: {tensor_shape}, "
                        f"found shape {input_tensor.shape[chunk_dim]}"
                    )

            if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
                raise ValueError(
                    f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                    f"size {chunk_size}"
                )

            num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

            # chunk input tensor into tuples
            input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
            # apply forward fn to every tuple
            output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
            # concatenate output at same dimension
            return jt.cat(output_chunks, dim=chunk_dim)

        return forward_fn(*input_tensors)
    
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        
# DONE?
class Attention(Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)
        self.pruned_heads = set()

    def prune_linear_layer(self, layer, index, dim: int = 0) -> nn.Linear:
        """
        Prune a linear layer to keep only entries in index.

        Used to remove heads.

        Args:
            layer (`jt.nn.Linear`): The layer to prune.
            index (`jt.int64`): The indices to keep in the layer.
            dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

        Returns:
            `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
        """
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer


    def find_pruneable_heads_and_indices(
            heads: List[int], 
            n_heads: int, 
            head_size: int, 
            already_pruned_heads: Set[int]
    ) -> Tuple[Set[int], jt.int64]:
        """
        Finds the heads and their indices taking :obj:`already_pruned_heads` into account.
        Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

        Returns:
            `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
            into account and the indices of rows/columns to keep in the layer weight.
        """
        mask = jt.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = jt.arange(len(mask))[mask].long()
        return heads, index

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = self.find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = self.prune_linear_layer(self.self.query, index)
        self.self.key = self.prune_linear_layer(self.self.key, index)
        self.self.value = self.prune_linear_layer(self.self.value, index)
        self.output.dense = self.prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def execute(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


# DONE? (should be, i need to run it)
class SelfAttention(Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_decoder = True # TODO: CHECK config.is_decoder
        self.rotary_value = config.rotary_value

    # reshape and permute the dims to prepare for the attention head splitting
    def transpose_for_scores(self, x: jt.Var):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def execute(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # rotary query
        query_layer = self.apply_rotary(query_layer, sinusoidal_pos)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos)
            if self.rotary_value:
                value_layer = self.apply_rotary(value_layer, sinusoidal_pos)
            key_layer = jt.concat([past_key_value[0], key_layer], dim=-2)
            value_layer = jt.concat([past_key_value[1], value_layer], dim=-2)

        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos)
            if self.rotary_value:
                value_layer = self.apply_rotary(value_layer, sinusoidal_pos)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = jt.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # if attention_mask is not None:  TODO
        #     print("ATTENTION MASK: ", attention_mask.shape)
        #     print("ATTENTION SCORES: ", attention_scores.shape)
        #     # Apply the attention mask
        #     attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context layer
        context_layer: jt.Var = jt.matmul(attention_probs, value_layer)

        context_layer = jt.reshape(context_layer.permute(0, 2, 1, 3), hidden_states.shape) # TODO: CHECK
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = jt.reshape(context_layer, *new_context_layer_shape) # TODO: CHECK

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        x = jt.stack([x1_rot, x2_rot], dim=-1)
        x = x.flatten(-2, -1)
        return x

# DONE
class SelfOutput(Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.norm_type == 'layer_norm' else Norm(eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def execute(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# DONE
class Output(Module):
    def __init__(self, config):
        super(Output, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.norm_type == 'layer_norm' else Norm(eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def execute(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# DONE
class Intermediate(Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.intermediate_act_fn = nn.GELU() # just using GELU instead of ACT2FN[config.hidden_act]

    def execute(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# DONE
class Norm(Module):
    def __init__(self, eps):
        super(Norm, self).__init__()
        self.eps = eps

    def execute(self, x):
        variance = jt.mean(x**2, dim=-1, keepdims=True)
        x = x * jt.rsqrt(variance + self.eps)
        return x

#####################################################

# Only MLM Head
class Cls(Module):
    def __init__(self, config):
        super(Cls, self).__init__()
        self.predictions = RoFormerLMPredictionHead(config) if config.norm_type=="layer_norm" else RoFormerV2LMPredictionHead(config)

    def execute(self, hidden_states):
        hidden_states = self.predictions(hidden_states)
        return hidden_states
    

class RoFormerV2LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.decoder(hidden_states)

class RoFormerLMPredictionHead(Module):
    def __init__(self, config):
        super(RoFormerLMPredictionHead, self).__init__()
        self.transform = PredictionHeadTransform(config)
        # Assuming the output dimension for the decoder matches the vocabulary size or embedding_num
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.bias = jt.zeros(config.vocab_size)

        self.decoder.bias = self.bias

    def execute(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class PredictionHeadTransform(Module):
    def __init__(self, config):
        super(PredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def execute(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class RoFormerPooler(Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = ACT2FN[config.pooler_activation]

    def execute(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        # pooled_output = self.activation(pooled_output)
        return pooled_output
