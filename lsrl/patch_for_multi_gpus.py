import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

# ---- Qwen3.5 adapted version ----
# Original (qwen2_original) hardcoded Qwen2's DecoderLayer forward signature.
# Qwen3.5 changes: (1) position_embeddings is the 2nd positional arg (was last in Qwen2)
#                   (2) layers have two types: linear_attention (GatedDeltaNet) and full_attention
#                   (3) fewer positional args (no output_attentions, use_cache, cache_position)

def decoding_layer_forward(self, hidden_states, position_embeddings,
        attention_mask=None, position_ids=None, past_key_values=None, **kwargs
    ):
    """PP-aware forward for Qwen3_5DecoderLayer.
    Moves tensors to the device where this layer's parameters live,
    then runs the original layer logic (handling both linear_attention and full_attention).
    """
    obj_device = next(self.parameters()).device

    # Move inputs to this layer's device (core of PP: layers live on different GPUs)
    if hidden_states.device != obj_device:
        hidden_states = hidden_states.to(obj_device)
    if position_embeddings[0].device != obj_device:
        position_embeddings = tuple(x.to(obj_device) for x in position_embeddings)
    if attention_mask is not None:
        attention_mask = attention_mask.to(obj_device)
    if position_ids is not None:
        position_ids = position_ids.to(obj_device)

    # --- The rest mirrors Qwen3_5DecoderLayer.forward exactly ---
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Token Mixer: two types of attention layers
    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    # MLP
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def chunked_lm_head_forward(self, input_ids, labels, use_cache=False, **kwargs):
    """Chunked lm_head to save VRAM: compute logits in small chunks instead of all at once."""
    # _text_model is set by patch function, works for both ForCausalLM and ForConditionalGeneration
    hidden = self._text_model(input_ids, use_cache=use_cache).last_hidden_state
    chunk_size = self._lm_head_chunk_size if hasattr(self, '_lm_head_chunk_size') else 1024
    loss_fct = nn.CrossEntropyLoss(reduction='sum')
    hiddens = hidden[:,:-1].reshape(-1, hidden.size(-1)).contiguous()
    labels = labels[:,1:].reshape(-1).contiguous()
    total_len = hiddens.size(0)
    real_len = labels.ne(-100).sum().item()
    for i in range(0, total_len, chunk_size):
        end = min(i + chunk_size, total_len)
        logits = self.lm_head(hiddens[i:end])
        sub_labels = labels[i:end].to(logits.device)
        loss = loss_fct(logits.float(), sub_labels)
        total_loss = loss if i == 0 else total_loss + loss
    loss = total_loss / real_len
    return CausalLMOutputWithPast(loss=loss)


def _find_text_model(model):
    """Auto-detect the text model inside different Qwen3.5 wrapper classes.
    - Qwen3_5ForCausalLM:              model.model (is Qwen3_5TextModel directly)
    - Qwen3_5ForConditionalGeneration:  model.model.language_model (Qwen3_5TextModel)
    Returns (text_model, layers, embed_tokens, norm).
    """
    inner = model.model  # first .model
    if hasattr(inner, 'language_model'):
        # ForConditionalGeneration -> Qwen3_5Model -> .language_model is TextModel
        text_model = inner.language_model
    elif hasattr(inner, 'layers'):
        # ForCausalLM -> .model is TextModel directly
        text_model = inner
    else:
        raise ValueError(f"Cannot find text model in {type(model)}. "
                         f"Expected Qwen3_5ForCausalLM or Qwen3_5ForConditionalGeneration.")
    return text_model


def patch_qwen2_for_multi_gpus(model, devices, patch_lm_head=False, chunk_size=1024):
    """Split a Qwen3.5 model across multiple GPUs using Pipeline Parallelism.
    Compatible with both Qwen3_5ForCausalLM and Qwen3_5ForConditionalGeneration.
    """
    text_model = _find_text_model(model)

    # Patch each decoder layer's forward to handle cross-device tensor movement
    for layer in text_model.layers:
        layer.forward = decoding_layer_forward.__get__(layer, type(layer))

    # Optionally patch lm_head to use chunked computation (saves VRAM for long sequences)
    if patch_lm_head:
        model._lm_head_chunk_size = chunk_size
        model._text_model = text_model  # store reference for chunked_lm_head_forward
        model.forward = chunked_lm_head_forward.__get__(model, type(model))

    # Build the list of "components" to distribute across GPUs:
    # [embed_tokens, rotary_emb, layer0, layer1, ..., layerN, norm, lm_head]
    # NOTE: Qwen3.5 has rotary_emb at TextModel level (not inside each layer like Qwen2)
    layers = [text_model.embed_tokens, text_model.rotary_emb]
    layers += list(text_model.layers)
    layers += [text_model.norm, model.lm_head]

    device_count = len(devices)
    print(f"Split model to {device_count} GPUs.")
    # Distribute layers as evenly as possible
    chunk_size_pp = (len(layers) + 3) // device_count + 1
    ids = [i // chunk_size_pp for i in range(len(layers) + 3)][2:-1]
    for layer, k in zip(layers, ids):
        layer.to(f'cuda:{devices[k]}')
    print(f'Make sure that the input should in cuda:{devices[0]} and the output in cuda:{devices[-1]}...')
