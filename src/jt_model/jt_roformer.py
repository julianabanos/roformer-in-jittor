import math
import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np
import json
import os

# jt.flags.use_cuda = 1

# params
# embedding_size = 768
# embedding_num = 50000
# intermediate_size = 3072

# roformer model

# cls
# predictions: transform, decoder

class Model(Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.roformer = Roformer(config)
        self.cls = Cls(config)

    def execute(self, input_ids, token_type_ids, attention_mask):
        # input_ids: (batch_size, seq_len)
        # token_type_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        roformer_output = self.roformer(input_ids, token_type_ids, attention_mask)
        cls_output = self.cls(roformer_output)
        return cls_output


#####################################################

class Roformer(Module):
    def __init__(self, config):
        super(Roformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

    def execute(self, input_ids, token_type_ids, attention_mask):
        # input_ids: (batch_size, seq_len)
        # token_type_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        return encoder_output
    
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

class Encoder(Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList([Layer(config) for _ in range(12)])

    def execute(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
    

class Layer(Module):
    def __init__(self, config):
        super(Layer, self).__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def execute(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    


class Attention(Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def execute(self, hidden_states, attention_mask):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


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
        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.reshape(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        
        @staticmethod
        def apply_rotary(x, sinusoidal_pos):
            sin, cos = sinusoidal_pos
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return jt.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(*x.shape) # TODO CHECK
        
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
            if attention_mask is not None:
                # Apply the attention mask
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            # mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # context layer
            context_layer = jt.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3).reshape(hidden_states.shape) # TODO: CHECK
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.reshape(*new_context_layer_shape) # TODO: CHECK

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            if self.is_decoder:
                outputs = outputs + (past_key_value,)
            return outputs

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
        
class Cls(Module):
    def __init__(self, config):
        super(Cls, self).__init__()
        self.predictions = ClsPredictions(config)

    def execute(self, hidden_states):
        hidden_states = self.predictions(hidden_states)
        return hidden_states
        
class ClsPredictions(Module):
    def __init__(self, config):
        super(ClsPredictions, self).__init__()
        self.transform = ClsTransform(config)
        # Assuming the output dimension for the decoder matches the vocabulary size or embedding_num
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.bias = jt.zeros(config.vocab_size)

    def execute(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class ClsTransform(Module):
    def __init__(self, config):
        super(ClsTransform, self).__init__()
        self.dense = nn.Linear(config.embedding_size, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.eps)

    def execute(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

