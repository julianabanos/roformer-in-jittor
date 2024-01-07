from dataclasses import dataclass

@dataclass
class DataClassEncoder:
    def __init__(self, 
                last_hidden_state,
                past_key_values,
                hidden_states,
                attentions,
                cross_attentions,
                ):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cross_attentions = cross_attentions

@dataclass
class DataClassRoformer:
    def __init__(self,
                last_hidden_state, # Expected: 1, 11, 768 Actual: 2, 753, 768
                pooler_output, # Expected: 1, 768 Actual: 2, 768
                past_key_values, # Expected: 1, 12, 11 64 Actual: 2, 12, 753, 64
                hidden_states, # Expected: None Actual: 2, 753, 768
                attentions, # Expected: None Actual: 2, 12, 753, 768
                cross_attentions, # Expected: None Actual: 2, 12, 753, 64
                ):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cross_attentions = cross_attentions


@dataclass
class DataClassCausalLM:
    def __init__(self,
                loss,
                logits,
                pooler_output,
                past_key_values,
                hidden_states,
                attentions,
                cross_attentions,
                ):
        self.loss = loss
        self.logits = logits
        self.pooler_output = pooler_output
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cross_attentions = cross_attentions

@dataclass
class DataClassSequenceClassifier:
        def __init__(self,
                        loss,
                        logits,
                        hidden_states,
                        attentions,
                        ):
                self.loss = loss
                self.logits = logits
                self.hidden_states = hidden_states
                self.attentions = attentions