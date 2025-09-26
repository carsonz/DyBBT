#!/usr/bin/env python3
"""
分析训练数据中input的最大长度
"""

import json
import random
from transformers import AutoTokenizer

MAX = 1000


def analyze_length():
    # 加载数据
    with open('train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    lengths = []
    max_length = 0
    max_sample = None
    
    # 遍历数据计算长度
    for sample in data[:]:  # 只分析前1000个样本以节省时间
        # 构造输入文本
        belief_state = json.dumps(sample['belief_state'], ensure_ascii=False)
        available_actions = json.dumps(sample['available_actions'], ensure_ascii=False)
        
        input_text = f"""
You are the fast, intuitive component (System 1) of a dialogue system. Your task is to generate the next system action based solely on the current belief state. Do not reason step-by-step. Output your first, most intuitive response in the exact JSON format specified.

**Current Belief State:**
{belief_state}

**Available Actions:**
{available_actions}

Based on the above, output ONLY a valid JSON object with your predicted action and its confidence. Do not output any other text.

{{"action": ["<act_type>", "<domain>", "<slot>"],"confidence": <confidence_score>}}
"""
        
        # 使用分词器编码input_text，设置max_length为3072
        encoded = tokenizer.encode(input_text, truncation=True, max_length=4096)
        lengths.append(len(encoded))
        
        # 保存最长的样本
        if len(encoded) > max_length:
            max_length = len(encoded)
            max_sample = input_text
    
    # 打印统计信息
    print(f"样本数量: {len(lengths)}")
    print(f"最大长度: {max_length}")
    print(f"平均长度: {sum(lengths) / len(lengths):.2f}")
    print(f"最小长度: {min(lengths)}")
    
    # 计算百分位数
    lengths.sort()
    print(f"90%分位数: {lengths[int(len(lengths) * 0.9)]}")
    print(f"95%分位数: {lengths[int(len(lengths) * 0.95)]}")
    print(f"99%分位数: {lengths[int(len(lengths) * 0.99)]}")
    
    # 显示最长样本的部分内容（编码后长度为3072）
    print("\n最长样本的部分内容（编码后长度为3072）:")
    # 使用分词器编码后截取3072个token
    if max_sample:
        encoded_sample = tokenizer.encode(max_sample, truncation=True, max_length=MAX)
        decoded_sample = tokenizer.decode(encoded_sample, skip_special_tokens=True)
        print(decoded_sample)


if __name__ == "__main__":
    analyze_length()