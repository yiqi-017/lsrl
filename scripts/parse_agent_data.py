#!/usr/bin/env python3
"""
Agent对话数据解析和转换工具
将model_responses.txt转换为SFT训练格式，支持试错轨迹筛选和错误截断
"""
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

@dataclass
class Turn:
    """单轮对话"""
    system: str = ""
    user: str = ""
    assistant: str = ""
    tool_result: Optional[str] = None
    has_error: bool = False
    error_type: str = ""

@dataclass
class Trajectory:
    """完整对话轨迹"""
    turns: List[Turn] = field(default_factory=list)
    user_goal: str = ""
    final_success: bool = False
    error_count: int = 0

    def has_trial_and_error(self) -> bool:
        """是否包含试错后成功的模式"""
        return self.error_count > 0 and self.final_success

def parse_model_responses(filepath: str) -> List[Trajectory]:
    """解析model_responses.txt文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    prompt_pattern = r'=== Prompt ===\s*=== SYSTEM ===\s*(.*?)\s*=== USER ===\s*(.*?)(?==== ASSISTANT ===|=== Prompt ===|$)'
    response_pattern = r'=== ASSISTANT ===\s*=== Response ===\s*(.*?)(?==== Prompt ===|$)'

    trajectories = []
    current_traj = Trajectory()

    sections = re.split(r'(?==== Prompt ===)', content)

    for section in sections:
        if not section.strip():
            continue

        turn = Turn()

        sys_match = re.search(r'=== SYSTEM ===\s*(.*?)(?==== USER ===|$)', section, re.DOTALL)
        if sys_match:
            turn.system = sys_match.group(1).strip()
            goal_match = re.search(r'# User Goal:\s*(.*?)(?=###|$)', turn.system, re.DOTALL)
            if goal_match and goal_match.group(1).strip():
                if not current_traj.user_goal:
                    current_traj.user_goal = goal_match.group(1).strip()

        user_match = re.search(r'=== USER ===\s*(.*?)(?==== ASSISTANT ===|$)', section, re.DOTALL)
        if user_match:
            turn.user = user_match.group(1).strip()
            if '<tool_result>' in turn.user:
                result_match = re.search(r'<tool_result>\s*(.*?)\s*</tool_result>', turn.user, re.DOTALL)
                if result_match:
                    turn.tool_result = result_match.group(1)
                    if any(err in turn.tool_result.lower() for err in
                           ['error', 'failed', 'failure', '失败', 'no such file', 'not found', 'exception']):
                        turn.has_error = True
                        turn.error_type = 'tool_error'
                        current_traj.error_count += 1

        resp_match = re.search(r'=== ASSISTANT ===\s*=== Response ===\s*(.*?)$', section, re.DOTALL)
        if resp_match:
            turn.assistant = resp_match.group(1).strip()

        if turn.system or turn.user or turn.assistant:
            current_traj.turns.append(turn)

        if is_new_task_boundary(turn, current_traj):
            if current_traj.turns:
                current_traj.final_success = check_final_success(current_traj)
                trajectories.append(current_traj)
            current_traj = Trajectory()

    if current_traj.turns:
        current_traj.final_success = check_final_success(current_traj)
        trajectories.append(current_traj)

    return trajectories


def is_new_task_boundary(turn: Turn, traj: Trajectory) -> bool:
    """判断是否是新任务的边界"""
    if not traj.turns:
        return False
    goal_match = re.search(r'# User Goal:\s*(.*?)(?=###|$)', turn.system, re.DOTALL)
    if goal_match:
        new_goal = goal_match.group(1).strip()
        if new_goal and new_goal != traj.user_goal and len(new_goal) > 10:
            return True
    return False


def check_final_success(traj: Trajectory) -> bool:
    """检查轨迹是否最终成功"""
    if not traj.turns:
        return False
    last_turns = traj.turns[-3:] if len(traj.turns) >= 3 else traj.turns
    for turn in reversed(last_turns):
        if turn.tool_result:
            if '"status": "success"' in turn.tool_result:
                if not any(err in turn.tool_result.lower() for err in
                          ['error', 'failed', 'exception', 'traceback']):
                    return True
        if turn.assistant:
            success_indicators = ['成功', '完成', 'successfully', 'done', '已保存', 'saved']
            if any(ind in turn.assistant.lower() for ind in success_indicators):
                return True
    return False


def truncate_error_trajectory(traj: Trajectory, keep_error_steps: int = 2) -> Trajectory:
    """截断错误轨迹，只保留前N步错误"""
    if traj.error_count <= keep_error_steps:
        return traj

    new_turns = []
    error_seen = 0
    in_error_sequence = False
    error_sequence_start = -1

    for i, turn in enumerate(traj.turns):
        if turn.has_error:
            if not in_error_sequence:
                in_error_sequence = True
                error_sequence_start = i
            error_seen += 1
            if error_seen <= keep_error_steps:
                new_turns.append(turn)
        else:
            if in_error_sequence and error_seen > keep_error_steps:
                pass
            new_turns.append(turn)
            in_error_sequence = False

    new_traj = Trajectory(
        turns=new_turns,
        user_goal=traj.user_goal,
        final_success=traj.final_success,
        error_count=min(traj.error_count, keep_error_steps)
    )
    return new_traj


def convert_to_sft_format(traj: Trajectory, include_system: bool = False) -> List[Dict]:
    """将轨迹转换为SFT训练格式"""
    samples = []
    for turn in traj.turns:
        if not turn.assistant:
            continue
        prompt_parts = []
        if include_system and turn.system:
            prompt_parts.append(turn.system)
        if turn.user:
            prompt_parts.append(turn.user)
        prompt = "\n\n".join(prompt_parts)
        if prompt and turn.assistant:
            samples.append({
                "instruction": prompt,
                "output": turn.assistant,
                "input": ""
            })
    return samples


def convert_to_multiturn_format(traj: Trajectory) -> Dict:
    """将轨迹转换为多轮对话格式"""
    messages = []
    for turn in traj.turns:
        if turn.system and not messages:
            messages.append({"role": "system", "content": turn.system})
        if turn.user:
            messages.append({"role": "user", "content": turn.user})
        if turn.assistant:
            messages.append({"role": "assistant", "content": turn.assistant})
    return {
        "messages": messages,
        "user_goal": traj.user_goal,
        "has_trial_and_error": traj.has_trial_and_error(),
        "error_count": traj.error_count,
        "final_success": traj.final_success
    }


def filter_trial_and_error_trajectories(
    trajectories: List[Trajectory],
    require_final_success: bool = True,
    min_errors: int = 1
) -> List[Trajectory]:
    """筛选试错后成功的轨迹"""
    filtered = []
    for traj in trajectories:
        if traj.error_count >= min_errors:
            if require_final_success and not traj.final_success:
                continue
            filtered.append(traj)
    return filtered


def process_and_save(
    input_file: str,
    output_dir: str,
    keep_error_steps: int = 2,
    filter_trial_error: bool = True,
    output_format: str = "sft"
):
    """处理数据并保存"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"解析文件: {input_file}")
    trajectories = parse_model_responses(input_file)
    print(f"解析到 {len(trajectories)} 条轨迹")

    stats = {
        "total": len(trajectories),
        "with_errors": sum(1 for t in trajectories if t.error_count > 0),
        "trial_and_error": sum(1 for t in trajectories if t.has_trial_and_error()),
        "final_success": sum(1 for t in trajectories if t.final_success)
    }
    print(f"统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

    if filter_trial_error:
        trajectories = filter_trial_and_error_trajectories(trajectories)
        print(f"筛选后剩余 {len(trajectories)} 条试错轨迹")

    processed = []
    for traj in trajectories:
        truncated = truncate_error_trajectory(traj, keep_error_steps)
        processed.append(truncated)

    if output_format == "sft":
        all_samples = []
        for traj in processed:
            samples = convert_to_sft_format(traj)
            all_samples.extend(samples)
        out_file = output_path / "sft_data.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        print(f"保存 {len(all_samples)} 条SFT样本到 {out_file}")
    else:
        all_convs = [convert_to_multiturn_format(t) for t in processed]
        out_file = output_path / "multiturn_data.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_convs, f, ensure_ascii=False, indent=2)
        print(f"保存 {len(all_convs)} 条多轮对话到 {out_file}")

    return processed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agent对话数据处理工具")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", default="./processed_data", help="输出目录")
    parser.add_argument("-k", "--keep-errors", type=int, default=2, help="保留的错误步数")
    parser.add_argument("--no-filter", action="store_true", help="不筛选试错轨迹")
    parser.add_argument("--format", choices=["sft", "multiturn"], default="sft")
    args = parser.parse_args()

    process_and_save(
        args.input,
        args.output,
        keep_error_steps=args.keep_errors,
        filter_trial_error=not args.no_filter,
        output_format=args.format
    )
