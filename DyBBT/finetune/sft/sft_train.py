#!/usr/bin/env python3
"""
Supervised Fine-Tuning Training Script
Implements LoRA SFT using Qwen/Qwen3-1.7B model and PEFT library
"""

import os
import json
import random
import numpy as np
import argparse
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from modelscope import snapshot_download


class SFTDataset(Dataset):
    """Supervised Fine-Tuning Dataset"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 833):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Construct input text
        belief_state = json.dumps(sample['belief_state'], ensure_ascii=False)
        available_actions = json.dumps(sample['available_actions'], ensure_ascii=False)
        
        # Construct dialogue format input
        input_text = [
            {"role": "system", "content": "You are the fast, intuitive component (System 1) of a dialogue system. Your task is to generate the next system action based solely on the current belief state. Do not reason step-by-step. Output your first, most intuitive response in the exact JSON format specified."},
            {"role": "user", "content": f"""
**Current Belief State:**
{belief_state}

**Available Actions:**
{available_actions}

Based on the above, output ONLY a valid JSON object with your predicted action and its confidence. Do not output any other text.

{{"action": [["<act_type>", "<domain>", "<slot>"], ["<act_type>", "<domain>", "<slot>"], ...],"confidence": <confidence_score>}}
"""}
        ]
        
        # Use all system actions as labels (supports multiple actions)
        if sample['system_action']:
            target_action = sample['system_action']
        else:
            # If no system action, use default action
            target_action = [["general", "bye", ""]]
        
        # Generate label with confidence score
        confidence = round(random.uniform(0.8, 0.99), 2)
        target_text = json.dumps({
            "action": target_action,
            "confidence": confidence
        }, ensure_ascii=False)
        
        # Encode input and labels
        # Convert dialogue format to model input format
        formatted_input = self.tokenizer.apply_chat_template(input_text, tokenize=False)
        input_ids = self.tokenizer.encode(formatted_input, truncation=True, max_length=self.max_length)
        target_ids = self.tokenizer.encode(target_text, truncation=True, max_length=self.max_length)
        
        # Construct model input
        input_ids = input_ids + target_ids
        labels = [-100] * len(input_ids[:-len(target_ids)]) + target_ids
        
        # Truncate to maximum length
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        
        # Pad to maximum length
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        labels.extend([-100] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Supervised Fine-Tuning Training Script')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-0.6B", 
                       help='Model name or path')
    parser.add_argument('--output_dir', type=str, default="./sft_checkpoints_Qwen3-0.6B", 
                       help='Output directory path')
    parser.add_argument('--train_data_path', type=str, default="train.json", 
                       help='Training data path')
    parser.add_argument('--val_data_path', type=str, default="val.json", 
                       help='Validation data path')
    parser.add_argument('--dataset_type', type=str, default="multiwoz", 
                       choices=['multiwoz', 'msdialog'],
                       help='Dataset type: multiwoz or msdialog')
    parser.add_argument('--domain', type=str, default="",
                       help='MSDialog domain name (movie/restaurant/taxi)')
    args = parser.parse_args()
    
    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # LLM-Research/Llama-3.2-1B-Instruct
    # LLM-Research/Llama-3.2-3B-Instruct
    # LLM-Research/gemma-2-2b-it
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0
    # bigcode/starcoder2-3b
    # tiiuae/falcon-7b-instruct
    # HuggingFaceH4/zephyr-7b-beta
    # mistralai/Mistral-7B-Instruct-v0.3
    # microsoft/Phi-3.5-mini-instruct
    # microsoft/Phi-4-mini-instruct
    # Qwen3-0.6B
    # Qwen3-1.7B
    # Qwen3-4B
    # Qwen3-8B
    # Qwen2.5-0.5B-Instruct
    # Qwen2.5-1.5B-Instruct
    # Qwen2.5-3B-Instruct
    # Qwen2.5-7B-Instruct
    
    # Configuration parameters
    model_name = args.model_name
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    output_dir = args.output_dir

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,               # Load model with 4-bit quantization
        bnb_4bit_quant_type="nf4",       # Quantization data type: NF4
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation, ensuring speed and stability
        bnb_4bit_use_double_quant=True,  # Use double quantization to further save memory
    )
    
    # Download model from ModelScope
    # model_dir = snapshot_download(model_name)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=bnb_config)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Print model information
    model.print_trainable_parameters()
    
    # Create datasets
    train_dataset = SFTDataset(train_data_path, tokenizer)
    val_dataset = SFTDataset(val_data_path, tokenizer)
    
    # Set training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,
        #optim="paged_adamw_8bit",        # Use paged 8bit AdamW optimizer to prevent memory peaks during gradient updates
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        num_train_epochs=5,
        learning_rate=1e-4,
        warmup_steps=500,
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to="tensorboard",
        save_total_limit=3,
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,  # This parameter is required for early stopping
        metric_for_best_model="eval_loss",  # Metric used for early stopping
        greater_is_better=False,  # Whether this metric is better when larger
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Add early stopping callback
    )
    
    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Supervised fine-tuning training completed!")

if __name__ == "__main__":
    main()