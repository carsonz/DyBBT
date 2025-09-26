#!/usr/bin/env python3
"""
为SFT模型添加价值头脚本
用于在SFT训练完成后为模型添加价值头，以便PPO训练使用
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def add_value_head_to_sft_model(sft_checkpoint_path, output_path=None):
    """
    为SFT模型添加价值头
    
    Args:
        sft_checkpoint_path: SFT模型检查点路径
        output_path: 输出路径，如果不提供则使用原路径
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_path is None:
        output_path = sft_checkpoint_path
    
    # 加载基础模型配置
    print(f"加载SFT模型从: {sft_checkpoint_path}")
    
    # 首先尝试加载基础模型
    try:
        # 检查是否有adapter_config.json文件
        if os.path.exists(os.path.join(sft_checkpoint_path, "adapter_config.json")):
            # 这是PEFT模型，需要先加载基础模型
            # 从adapter_config.json获取基础模型名称
            import json
            with open(os.path.join(sft_checkpoint_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "LLM-Research/Llama-3.2-1B-Instruct")
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
            
            # 加载PEFT适配器
            model = PeftModel.from_pretrained(base_model, sft_checkpoint_path)
            print(f"已加载PEFT模型，基础模型: {base_model_name}")
        else:
            # 直接加载完整模型
            model = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path, trust_remote_code=True)
            print("已加载完整模型")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None
    
    # 获取隐藏层大小
    hidden_size = model.config.hidden_size
    
    # 添加价值头
    if not hasattr(model, 'value_head'):
        model.value_head = nn.Linear(hidden_size, 1).to(DEVICE)
        print(f"价值头已添加到模型，输入维度: {hidden_size}")
    
    # 保存价值头权重
    os.makedirs(output_path, exist_ok=True)
    value_head_path = os.path.join(output_path, "value_head.pt")
    torch.save(model.value_head.state_dict(), value_head_path)
    print(f"价值头权重已保存到: {value_head_path}")
    
    # 如果输出路径不同，保存整个模型
    if output_path != sft_checkpoint_path:
        model.save_pretrained(output_path)
        print(f"完整模型已保存到: {output_path}")
    
    return model


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="为SFT模型添加价值头")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                       help="SFT模型检查点路径")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出路径，如果不提供则使用原路径")
    
    args = parser.parse_args()
    
    # 添加价值头
    model = add_value_head_to_sft_model(args.sft_checkpoint, args.output_path)
    
    if model is not None:
        print("价值头添加完成！")
        print("现在可以使用这个模型进行PPO训练了。")
    else:
        print("添加价值头失败！")


if __name__ == "__main__":
    main()