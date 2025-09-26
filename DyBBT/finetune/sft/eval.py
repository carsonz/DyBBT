#!/usr/bin/env python3
"""
模型评估脚本
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from DyBBT.utils import parse_model_output


def load_test_data(data_path: str):
    """加载测试数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_input(sample):
    """格式化输入文本"""
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
    return input_text

def evaluate_model(model, tokenizer, test_data, device, batch_size=8):
    """评估模型"""
    
    # 按domain统计结果
    domain_stats = {}
    total_correct = 0
    total_samples = 0
    fine_acc = 0
    total_fine_samples = 0
    
    # 分批处理数据
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating", unit="batch"):
        batch = test_data[i:i+batch_size]
        
        # 准备批次输入
        input_texts = [format_input(sample) for sample in batch]
        inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True, max_length=833).to(device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=1.0, top_k=0, top_p=1.0)
        
        # 解码输出
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 处理每个样本的结果
        for j, sample in enumerate(batch):
            # 获取真实标签（actions数组）
            true_actions = sample['system_action'] if sample['system_action'] else [["general", "bye", ""]]
            
            # 获取domain（取第一个动作的domain）
            domain = true_actions[0][1] if len(true_actions[0]) > 1 else "general"
            
            # 初始化domain统计
            if domain not in domain_stats:
                domain_stats[domain] = {'correct': 0, 'total': 0, 'fine_correct': 0, 'fine_total': 0}
            
            # 解析预测结果
            pred_actions = parse_model_output(output_texts[j])
            
            # 比较结果
            if pred_actions:
                # 完全匹配（fine accuracy）：所有动作完全一致
                if pred_actions == true_actions:
                    domain_stats[domain]['fine_correct'] += 1
                    fine_acc += 1
                
                # 部分匹配（coarse accuracy）：统计正确动作的数量
                correct_count = 0
                for pred_action in pred_actions:
                    if pred_action in true_actions:
                        correct_count += 1
                
                if correct_count > 0:
                    domain_stats[domain]['correct'] += correct_count
                    total_correct += correct_count
            
            # 统计总动作数
            domain_stats[domain]['total'] += len(true_actions)
            total_samples += len(true_actions)
            
            # 统计细粒度样本数（每个样本计为1）
            domain_stats[domain]['fine_total'] += 1
            total_fine_samples += 1
    
    return domain_stats, total_correct, total_samples, fine_acc, total_fine_samples


def print_results(domain_stats, total_correct, total_samples, fine_acc, total_fine_samples):
    """打印评估结果"""
    print("\n=== 评估结果 ===")
    
    # 按domain打印准确率
    for domain, stats in domain_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        fine_accuracy = stats['fine_correct'] / stats['fine_total'] if stats['fine_total'] > 0 else 0
        print(f"{domain}: 粗粒度准确率 {accuracy:.4f} ({stats['correct']}/{stats['total']}), 细粒度准确率 {fine_accuracy:.4f} ({stats['fine_correct']}/{stats['fine_total']})")
    
    # 打印总平均准确率
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    fine_accuracy = fine_acc / total_fine_samples if total_fine_samples > 0 else 0
    print(f"\n总粗粒度准确率: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
    print(f"总细粒度准确率: {fine_accuracy:.4f} ({fine_acc}/{total_fine_samples})")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='qwen', help='Model type: qwen or sft')
    parser.add_argument('--model_path', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Model path')
    parser.add_argument('--adapter_path', type=str, default='../sft_checkpoints-TinyLlama-1.1B-Chat-v1.0', help='Adapter path for SFT model')
    parser.add_argument('--test_data', type=str, default='test.json', help='Test data path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.model_type == 'qwen':
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    elif args.model_type == 'sft':
        # 加载SFT模型
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to(device)
        model = PeftModel.from_pretrained(model, args.adapter_path).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 加载测试数据
    test_data = load_test_data(args.test_data)
    print(f"加载了 {len(test_data)} 个测试样本")
    
    # 评估模型
    domain_stats, total_correct, total_samples, fine_acc, total_fine_samples = evaluate_model(model, tokenizer, test_data, device, batch_size=args.batch_size)
    
    # 打印结果
    print_results(domain_stats, total_correct, total_samples, fine_acc, total_fine_samples)


if __name__ == "__main__":
    main()