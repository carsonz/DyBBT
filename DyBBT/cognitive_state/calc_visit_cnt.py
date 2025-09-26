#!/usr/bin/env python3
"""
计算访问认知空间的计数

认知状态是 c_t = [d_t, u_t, rho_t]
根据要求，将连续认知状态空间离散化为5个bins，计算访问计数
"""

import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def load_slot_cooccurrence_matrix(matrix_path: str) -> Dict:
    """加载槽位共现矩阵"""
    with open(matrix_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_slot_dependency(belief_state: Dict, cooccurrence_matrix: Dict) -> float:
    """
    计算槽位依赖度 rho_t
    
    Args:
        belief_state: 当前的信念状态
        cooccurrence_matrix: 槽位共现矩阵
    
    Returns:
        槽位依赖度 rho_t
    """
    # 提取已知槽位（有值的槽位）和未知槽位（无值的槽位）
    known_slots = []
    unknown_slots = []
    
    for domain, slots in belief_state.items():
        for slot, value in slots.items():
            full_slot_name = f"{domain}-{slot}"
            if value and value.strip():  # 有值的槽位
                known_slots.append(full_slot_name)
            else:  # 无值的槽位
                unknown_slots.append(full_slot_name)
    
    # 如果没有已知槽位或未知槽位，返回0
    if not known_slots or not unknown_slots:
        return 0.0
    
    # 计算每个未知槽位对已知槽位的依赖度
    dependency_scores = {}
    matrix = cooccurrence_matrix['cooccurrence_matrix']
    
    for u in unknown_slots:
        if u in matrix:
            total_prob = 0
            count = 0
            for f in known_slots:
                if f in matrix[u]:
                    total_prob += matrix[u][f]
                    count += 1
            if count > 0:
                avg_prob = total_prob / count
                dependency_scores[u] = avg_prob
    
    # 返回最大依赖度
    return max(dependency_scores.values()) if dependency_scores else 0.0


def calculate_progress(belief_state: Dict) -> float:
    """
    计算进度 d_t
    
    Args:
        belief_state: 当前的信念状态
    
    Returns:
        进度值 d_t，范围 [0, 1]
    """
    total_slots = 0
    filled_slots = 0
    
    for domain, slots in belief_state.items():
        for slot, value in slots.items():
            total_slots += 1
            if value and value.strip():  # 有值的槽位
                filled_slots += 1
    
    if total_slots == 0:
        return 0.0
    
    return filled_slots / total_slots


def calculate_uncertainty(belief_state: Dict, available_actions: List) -> float:
    """
    计算不确定性 u_t
    
    Args:
        belief_state: 当前的信念状态
        available_actions: 可用动作列表
    
    Returns:
        不确定性值 u_t，范围 [0, 1]
    """
    # 简单的不确定性计算：基于可用动作的数量
    # 动作越多，不确定性越高
    if not available_actions:
        return 0.0
    
    # 假设最大动作数为50（根据观察数据调整）
    max_actions = 50
    return min(len(available_actions) / max_actions, 1.0)


def discretize_cognitive_state(d_t: float, u_t: float, rho_t: float, num_bins: int = 5) -> Tuple[int, int, int]:
    """
    将连续认知状态离散化为bins
    
    Args:
        d_t: 进度值 [0, 1]
        u_t: 不确定性值 [0, 1]
        rho_t: 槽位依赖度 [0, 1]
        num_bins: bins数量，默认为5
    
    Returns:
        离散化的bin元组 (d_bin, u_bin, rho_bin)
    """
    # 确保值在[0, 1]范围内
    d_t = max(0.0, min(1.0, d_t))
    u_t = max(0.0, min(1.0, u_t))
    rho_t = max(0.0, min(1.0, rho_t))
    
    # 将[0, 1]范围离散化为num_bins个bins
    d_bin = min(int(d_t * num_bins), num_bins - 1)
    u_bin = min(int(u_t * num_bins), num_bins - 1)
    rho_bin = min(int(rho_t * num_bins), num_bins - 1)
    
    return (d_bin, u_bin, rho_bin)


def process_dialogue_data(train_data_path: str, cooccurrence_matrix_path: str, output_path: str):
    """
    处理对话数据并计算访问计数
    
    Args:
        train_data_path: 训练数据路径
        cooccurrence_matrix_path: 槽位共现矩阵路径
        output_path: 输出文件路径
    """
    # 加载训练数据
    print("加载训练数据...")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 加载槽位共现矩阵
    print("加载槽位共现矩阵...")
    cooccurrence_matrix = load_slot_cooccurrence_matrix(cooccurrence_matrix_path)
    
    # 初始化访问计数字典
    visit_count = defaultdict(int)
    
    # 处理每个对话样本
    print("处理对话数据并计算访问计数...")
    for i, sample in enumerate(train_data):
        if i % 1000 == 0:
            print(f"已处理 {i} 个样本...")
        
        dialog_id = sample['dialog_id']
        belief_state = sample['belief_state']
        available_actions = sample['available_actions']
        
        # 计算认知状态的三个维度
        d_t = calculate_progress(belief_state)
        u_t = calculate_uncertainty(belief_state, available_actions)
        rho_t = calculate_slot_dependency(belief_state, cooccurrence_matrix)
        
        # 离散化认知状态
        bin_tuple = discretize_cognitive_state(d_t, u_t, rho_t)
        
        # 累积访问计数
        visit_count[str(bin_tuple)] += 1
    
    # 保存结果
    print("保存访问计数结果...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(visit_count, f, indent=2, ensure_ascii=False)
    
    print(f"访问计数已保存到: {output_path}")
    
    # 打印一些统计信息
    print(f"\n统计信息:")
    print(f"  - 总共处理样本数: {len(train_data)}")
    print(f"  - 不同的bin元组数: {len(visit_count)}")
    print(f"  - 最大访问计数: {max(visit_count.values())}")
    print(f"  - 平均访问计数: {np.mean(list(visit_count.values())):.2f}")


def main():
    """主函数"""
    # 设置路径
    project_root = "/home/zsy/workspace/gitee/DyBBT/DyBBT"
    train_data_path = os.path.join(project_root, "finetune", "sft", "train.json")
    cooccurrence_matrix_path = os.path.join(project_root, "slot_comatrix", "slot_cooccurrence_matrix.json")
    output_path = os.path.join(project_root, "cognitive_state", "visit_cnt.json")
    
    # 处理数据
    process_dialogue_data(train_data_path, cooccurrence_matrix_path, output_path)


if __name__ == "__main__":
    main()