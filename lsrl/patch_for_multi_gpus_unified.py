import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

# ---- Qwen2 version ----
def decoding_layer_forward_qwen2(self, hidden_states, attention_mask=None,
        position_ids=None, past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None, position_embeddings=None, **kwargs
    ):
        obj_device = next(self.parameters()).device
        if hidden_states.device != obj_device:
            hidden_states = hidden_states.to(obj_device)
        if position_embeddings is not None and position_embeddings[0].device != obj_device:
            position_embeddings = [x.to(obj_device) for x in position_embeddings]
        if attention_mask is not None: attention_mask = attention_mask.to(obj_device)
        if position_ids is not None: position_ids = position_ids.to(obj_device)
        if cache_position is not None: cache_position = cache_position.to(obj_device)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights,)
        return outputs

# ---- Qwen3.5 version ----
def decoding_layer_forward_qwen3(self, hidden_states, position_embeddings,
        attention_mask=None, position_ids=None, past_key_values=None, **kwargs
    ):
    obj_device = next(self.parameters()).device
    if hidden_states.device != obj_device:
        hidden_states = hidden_states.to(obj_device)
    if position_embeddings[0].device != obj_device:
        position_embeddings = tuple(x.to(obj_device) for x in position_embeddings)
    if attention_mask is not None:
        attention_mask = attention_mask.to(obj_device)
    if position_ids is not None:
        position_ids = position_ids.to(obj_device)

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

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
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def chunked_lm_head_forward_qwen2(self, input_ids, labels, use_cache=False, **kwargs):
    hidden = self.model(input_ids, use_cache=use_cache).last_hidden_state
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


def chunked_lm_head_forward_qwen3(self, input_ids, labels, use_cache=False, **kwargs):
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


def _detect_model_type(model):
    """检测模型类型: qwen2 或 qwen3"""
    model_type = getattr(model.config, 'model_type', '')
    if 'qwen3' in model_type.lower() or 'qwen3_5' in model_type.lower():
        return 'qwen3'
    return 'qwen2'


def _find_text_model_qwen3(model):
    """为Qwen3.5找到text model"""
    inner = model.model
    if hasattr(inner, 'language_model'):
        return inner.language_model
    elif hasattr(inner, 'layers'):
        return inner
    else:
        raise ValueError(f"Cannot find text model in {type(model)}")


def patch_qwen2_for_multi_gpus(model, devices, patch_lm_head=False, chunk_size=1024):
    """统一的multi-GPU patch函数，自动检测模型类型"""
    model_type = _detect_model_type(model)
    print(f"Detected model type: {model_type}")

    if model_type == 'qwen3':
        text_model = _find_text_model_qwen3(model)
        for layer in text_model.layers:
            layer.forward = decoding_layer_forward_qwen3.__get__(layer, type(layer))
        if patch_lm_head:
            model._lm_head_chunk_size = chunk_size
            model._text_model = text_model
            model.forward = chunked_lm_head_forward_qwen3.__get__(model, type(model))
        layers = [text_model.embed_tokens, text_model.rotary_emb]
        layers += list(text_model.layers)
        layers += [text_model.norm, model.lm_head]
    else:
        for layer in model.model.layers:
            layer.forward = decoding_layer_forward_qwen2.__get__(layer, nn.Module)
        if patch_lm_head:
            model._lm_head_chunk_size = chunk_size
            model.forward = chunked_lm_head_forward_qwen2.__get__(model, nn.Module)
        layers = [model.model.embed_tokens]
        layers += [x for x in model.model.layers]
        layers += [model.model.norm, model.lm_head]

    device_count = len(devices)
    print(f"Split model to {device_count} GPUs.")
    chunk_size_pp = (len(layers) + 3) // device_count + 1
    ids = [i // chunk_size_pp for i in range(len(layers) + 3)][2:-1]
    for layer, k in zip(layers, ids):
        layer.to(f'cuda:{devices[k]}')
    print(f'Make sure that the input should in cuda:{devices[0]} and the output in cuda:{devices[-1]}...')
