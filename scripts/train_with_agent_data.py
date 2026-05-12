#!/usr/bin/env python3
"""
使用Agent对话数据进行SFT训练
支持试错轨迹筛选和错误截断
"""
import os
import sys
import json
import time
import argparse

os.environ['OMP_NUM_THREADS'] = '32'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsrl import CPUAdamW
from lsrl.patch_for_multi_gpus_qwen2 import patch_qwen2_for_multi_gpus
from lsrl.dataloader import SFTDataHandler

from parse_agent_data import process_and_save, parse_model_responses
from parse_agent_data import filter_trial_and_error_trajectories, truncate_error_trajectory
from parse_agent_data import convert_to_sft_format


def load_and_process_data(
    input_file: str,
    keep_error_steps: int = 2,
    filter_trial_error: bool = True,
    include_all: bool = False
):
    """加载并处理Agent对话数据"""
    print(f"解析文件: {input_file}")
    trajectories = parse_model_responses(input_file)
    print(f"解析到 {len(trajectories)} 条轨迹")

    total_errors = sum(t.error_count for t in trajectories)
    trial_error_count = sum(1 for t in trajectories if t.has_trial_and_error())
    print(f"总错误数: {total_errors}, 试错后成功轨迹: {trial_error_count}")

    if filter_trial_error and not include_all:
        trajectories = filter_trial_and_error_trajectories(trajectories)
        print(f"筛选后剩余 {len(trajectories)} 条试错轨迹")

    processed = []
    for traj in trajectories:
        truncated = truncate_error_trajectory(traj, keep_error_steps)
        processed.append(truncated)

    all_samples = []
    for traj in processed:
        samples = convert_to_sft_format(traj)
        all_samples.extend(samples)

    print(f"生成 {len(all_samples)} 条SFT训练样本")
    return all_samples


def main():
    parser = argparse.ArgumentParser(description="使用Agent对话数据进行SFT训练")
    parser.add_argument("--data", required=True, help="输入数据文件路径")
    parser.add_argument("--model", default="/mnt/data/kw/models/Qwen2.5-14B-Instruct",
                        help="模型路径")
    parser.add_argument("--keep-errors", type=int, default=2, help="保留的错误步数")
    parser.add_argument("--no-filter", action="store_true", help="不筛选试错轨迹")
    parser.add_argument("--include-all", action="store_true", help="包含所有数据")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--accum-steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--max-len", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--steps", type=int, default=100, help="训练步数")
    parser.add_argument("--save-path", default=None, help="模型保存路径")
    args = parser.parse_args()

    if 'RANK' in os.environ:
        print("\nError: This script must be run with python, not torchrun")
        print(f"Example: CUDA_VISIBLE_DEVICES=0,1,2,3 python {__file__} --data <path>")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("Agent对话数据SFT训练")
    print(f"{'='*50}")

    data = load_and_process_data(
        args.data,
        keep_error_steps=args.keep_errors,
        filter_trial_error=not args.no_filter,
        include_all=args.include_all
    )

    if not data:
        print("没有可用的训练数据!")
        sys.exit(1)

    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable()

    device_count = torch.cuda.device_count()
    devices = list(range(device_count))
    print(f"使用 {device_count} 个GPU: {devices}")
    patch_qwen2_for_multi_gpus(model, devices=devices, patch_lm_head=True)

    opt = CPUAdamW(model.parameters(), lr=args.lr, accum_steps=args.accum_steps,
                   weight_decay=0.01, eps=1e-8, grad_offload=True)

    handler = SFTDataHandler(data, tokenizer, max_len=args.max_len, mode='packing')
    dataloader = handler.get_dataloader(shuffle=True)

    print(f"\n开始训练, 共 {args.steps} 步...")
    step = 0
    total_loss = 0
    for epoch in range(100):
        for batch in dataloader:
            step += 1
            if step > args.steps:
                break

            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)

            tic = time.time()
            loss = model(input_ids, labels=labels, use_cache=False).loss
            loss.backward()
            total_loss += loss.item()

            updated = opt.step()
            elapsed = time.time() - tic

            if step % 10 == 0 or updated:
                avg_loss = total_loss / step
                print(f"Step {step}: loss={loss.item():.4f}, avg={avg_loss:.4f}, "
                      f"time={elapsed:.2f}s" + (" [updated]" if updated else ""))

        if step > args.steps:
            break

    print(f"\n训练完成! 平均loss: {total_loss/step:.4f}")

    if args.save_path:
        print(f"保存模型到: {args.save_path}")
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)

    print("Done!")


if __name__ == "__main__":
    main()
