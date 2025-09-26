#!/usr/bin/env python3
"""
数据预处理脚本
支持MultiWOZ和MSDialog三个领域数据集的处理
每个对话样本包含：
- belief_state: 每一轮的信念状态（核心输入）。
- system_action: 专家（或基线模型）执行的系统动作（即训练标签）。
- domain： 从user_action中提取的领域（一个或多个）。
- available_actions： 系统在当前状态下可选的动作（根据领域和信念状态）。
- dialog_id: 对话ID，格式为原对话ID_对话轮次
- user_action: 用户动作
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Set

# 添加convlab到Python路径
sys.path.append('/home/zsy/workspace/gitee/DyBBT')

from convlab.util.unified_datasets_util import load_dataset
from convlab.policy.vector.vector_base import VectorBase
from DyBBT.utils import extract_domains_from_action, get_available_actions, CustomVectorBase

def load_local_dataset(dataset_name: str) -> Dict:
    """直接从本地文件加载数据集，避免网络下载"""
    data_dir = f'/home/zsy/workspace/gitee/DyBBT/data/unified_datasets/{dataset_name}'
    
    # 检查dialogues.json文件是否存在
    dialogues_path = os.path.join(data_dir, 'dialogues.json')
    if not os.path.exists(dialogues_path):
        # 如果本地文件不存在，回退到原始方法
        return load_dataset(dataset_name)
    
    # 从本地文件加载数据
    with open(dialogues_path, 'r', encoding='utf-8') as f:
        dialogues_data = json.load(f)
    
    # 处理不同的数据格式
    if isinstance(dialogues_data, dict) and 'train' in dialogues_data:
        # MSDialog格式：顶层是数据分割键
        dataset = dialogues_data
    elif isinstance(dialogues_data, list):
        # MultiWOZ格式：对话列表，需要按数据分割组织
        dataset = {}
        for dialogue in dialogues_data:
            data_split = dialogue.get('data_split', 'train')
            if data_split not in dataset:
                dataset[data_split] = []
            dataset[data_split].append(dialogue)
    else:
        raise ValueError(f"Unknown dataset format for {dataset_name}")
    
    return dataset

def flatten_system_action(dialogue_acts: Dict) -> List[List]:
    """将系统动作从categorical、non-categorical、binary分类转换为直接的动作列表"""
    flattened_actions = []
    
    # 遍历所有动作类型（categorical, non-categorical, binary）
    for act_type in ['categorical', 'non-categorical', 'binary']:
        if act_type in dialogue_acts:
            # 遍历该类型下的所有动作
            for act in dialogue_acts[act_type]:
                # 提取动作的domain、intent、slot
                domain = act.get('domain', '')
                intent = act.get('intent', '')
                slot = act.get('slot', '')
                value = act.get('value', '')
                
                # 处理value字段：如果为"?"则保留，否则设为空字符串
                if value != "?":
                    value = ""
                
                # 构造动作列表，只保留domain、intent、slot
                action_list = [domain, intent, slot]
                flattened_actions.append(action_list)
    
    return flattened_actions

def process_multiwoz_dataset(dataset, data_split: str, vector_base: VectorBase) -> List[Dict]:
    """处理MultiWOZ数据集"""
    processed_data = []
    
    for dialogue in dataset[data_split]:
        turns = dialogue['turns']
        dialog_id = dialogue['dialogue_id']
        # 遍历所有用户回合（偶数索引）
        for i in range(0, len(turns), 2):
            # 确保这是用户回合
            if turns[i]['speaker'] == 'user':
                # 提取用户动作
                user_action = turns[i].get('dialogue_acts', {})
                # 展平用户动作
                flattened_user_action = flatten_system_action(user_action)
                
                # 提取系统动作（下一个系统回合）
                system_action = {}
                if i+1 < len(turns) and turns[i+1]['speaker'] == 'system':
                    system_action = turns[i+1].get('dialogue_acts', {})
                
                # 展平系统动作
                flattened_system_action = flatten_system_action(system_action)
                
                # 提取信念状态（从当前用户回合中获取）
                belief_state = {}
                full_belief_state = turns[i].get('state', {})
                # 仅保留与domain相关的信念状态
                domain = extract_domains_from_action(turns[i].get('dialogue_acts', {}))
                for d in domain:
                    if d in full_belief_state:
                        belief_state[d] = full_belief_state[d]
                
                # 获取可用动作（仅与domain相关的动作）
                available_actions = get_available_actions(vector_base, domain, belief_state)
                
                # 构建样本
                sample = {
                    'dialog_id': f"{dialog_id}_{i}",  # 添加对话ID
                    'domain': domain,
                    'belief_state': belief_state,
                    'system_action': flattened_system_action,  # 使用展平后的系统动作
                    #'user_action': flattened_user_action,  # 添加用户动作
                    'available_actions': available_actions
                }
                
                processed_data.append(sample)
    
    return processed_data

def process_msdialog_dataset(dataset, data_split: str, vector_base: VectorBase, domain_name: str) -> List[Dict]:
    """处理MSDialog数据集"""
    processed_data = []
    
    for dialogue in dataset[data_split]:
        turns = dialogue['turns']
        dialog_id = dialogue['dialogue_id']
        
        # MSDialog数据通常只有一轮对话
        if len(turns) > 0 and turns[0]['speaker'] == 'user':
            # 提取用户动作（简化处理）
            user_utterance = turns[0].get('utterance', '')
            
            # 提取信念状态
            belief_state = turns[0].get('state', {}).get(domain_name, {})
            
            # 提取系统动作（如果有）
            system_action = {}
            if len(turns) > 1 and turns[1]['speaker'] == 'system':
                system_action = turns[1].get('dialogue_acts', {})
            
            # 展平系统动作
            flattened_system_action = flatten_system_action(system_action)
            
            # 获取可用动作
            available_actions = get_available_actions(vector_base, [domain_name], {domain_name: belief_state})
            
            # 构建样本
            sample = {
                'dialog_id': dialog_id,
                'domain': [domain_name],
                'belief_state': {domain_name: belief_state},
                'system_action': flattened_system_action,
                'available_actions': available_actions
            }
            
            processed_data.append(sample)
    
    return processed_data

def save_processed_data(data: List[Dict], output_path: str):
    """保存处理后的数据"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_dataset(dataset_name: str, data_split: str, vector_base: VectorBase) -> List[Dict]:
    """处理指定数据集"""
    # 使用本地数据集加载
    dataset = load_local_dataset(dataset_name)
    
    if dataset_name.startswith('multiwoz'):
        return process_multiwoz_dataset(dataset, data_split, vector_base)
    elif dataset_name.startswith('msdialog'):
        domain_name = dataset_name.split('_')[1]  # 提取领域名称
        return process_msdialog_dataset(dataset, data_split, vector_base, domain_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def main():
    """主函数"""
    print("开始处理数据集...")
    
    # 初始化CustomVectorBase
    vector_base = CustomVectorBase()
    
    # 定义要处理的数据集
    datasets = [
        'multiwoz21',
        'msdialog_movie',
        'msdialog_restaurant', 
        'msdialog_taxi'
    ]
    
    for dataset_name in datasets:
        print(f"处理数据集: {dataset_name}")
        
        try:
            # 处理train数据集
            print("处理train数据集...")
            train_data = process_dataset(dataset_name, 'train', vector_base)
            save_processed_data(train_data, f'{dataset_name}_train.json')
            
            # 处理validation数据集
            print("处理validation数据集...")
            val_data = process_dataset(dataset_name, 'validation', vector_base)
            save_processed_data(val_data, f'{dataset_name}_val.json')
            
            # 处理test数据集（如果有）
            if dataset_name != 'msdialog_taxi':  # taxi没有test集
                print("处理test数据集...")
                test_data = process_dataset(dataset_name, 'test', vector_base)
                save_processed_data(test_data, f'{dataset_name}_test.json')
            
            print(f"{dataset_name} 处理完成！")
            
        except Exception as e:
            print(f"处理 {dataset_name} 时出错: {e}")
            continue
    
    print("所有数据集处理完成！")

if __name__ == "__main__":
    main()