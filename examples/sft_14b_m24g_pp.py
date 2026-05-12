import os, time, sys
os.environ['OMP_NUM_THREADS'] = '32'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if 'RANK' in os.environ:
    print("\nError: This script must be run with python, not torchrun")
    print(f"Example usage: CUDA_VISIBLE_DEVICES=0,1,2,3 python {__file__}")
    sys.exit(1)

# ---- Changed: Qwen3.6-27B (model_type=qwen3_5) ----
# AutoModelForCausalLM will resolve to Qwen3_5ForCausalLM (not ForConditionalGeneration),
# which has the same .model.layers/.model.embed_tokens/.model.norm structure as Qwen2.
# This avoids the extra nesting from the multimodal wrapper.
model_path = "/mnt/data/kw/models/Huihui-Qwen3.6-27B-abliterated"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model.train()
model.gradient_checkpointing_enable()

from lsrl import CPUAdamW, patch_qwen2_for_multi_gpus

device_count = torch.cuda.device_count()
devices = list(range(device_count))
patch_qwen2_for_multi_gpus(model, devices=devices, patch_lm_head=True)

opt = CPUAdamW(model.parameters(), lr=1e-5, accum_steps=4,
               weight_decay=0.01, eps=1e-8, 
               grad_offload=True)

for step in range(1, 8):
    batch = torch.randint(1, 10, (2, 8500)).to(model.device)
    print('\nInput shape:', batch.shape)
    tic = time.time()
    loss = model(batch, labels=batch, use_cache=False).loss
    loss.backward()
    print('step: ', step, 'loss: %.4f' % loss.item())
    print('step time: ', end='')
    if opt.step(): print('update parameters! ')
    print('%.2fs' % (time.time()-tic))

print('\nNo OOM, Good!')
