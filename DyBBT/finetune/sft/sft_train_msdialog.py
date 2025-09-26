#!/usr/bin/env python3
"""
MSDialog Supervised Fine-tuning Training Script - Adapted for MSDialog three domains
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    BitsAndBytesConfig, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType


class MSDialogSFTDataset(Dataset):
    """MSDialog Supervised Fine-tuning Dataset Class"""
    
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
        
        # Use all system actions as labels (support multiple actions)
        if sample['system_action']:
            target_action = sample['system_action']
        else:
            # If no system action, use default action
            target_action = [["general", "bye", ""]]
        
        # Generate label with confidence
        confidence = round(random.uniform(0.8, 0.99), 2)
        target_text = json.dumps({
            "action": target_action,
            "confidence": confidence
        }, ensure_ascii=False)
        
        # Encode input and labels
        formatted_input = self.tokenizer.apply_chat_template(input_text, tokenize=False)
        input_ids = self.tokenizer.encode(formatted_input, truncation=True, max_length=self.max_length)
        target_ids = self.tokenizer.encode(target_text, truncation=True, max_length=self.max_length)
        
        # Construct model input
        input_ids = input_ids + target_ids
        labels = [-100] * len(input_ids[:-len(target_ids)]) + target_ids
        
        # Truncate to max length
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        
        # Pad to max length
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
    parser = argparse.ArgumentParser(description='MSDialog Supervised Fine-tuning Training Script')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-0.6B", 
                       help='Model name or path')
    parser.add_argument('--output_dir', type=str, default="./sft_checkpoints_msdialog", 
                       help='Output directory path')
    parser.add_argument('--train_data_path', type=str, required=True,
                       help='Training data path (e.g., msdialog_movie_train.json)')
    parser.add_argument('--val_data_path', type=str, required=True,
                       help='Validation data path (e.g., msdialog_movie_val.json)')
    parser.add_argument('--domain', type=str, required=True,
                       choices=['movie', 'restaurant', 'taxi'],
                       help='MSDialog domain name')
    args = parser.parse_args()
    
    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure parameters
    model_name = args.model_name
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    output_dir = f"{args.output_dir}_{args.domain}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,               # Load model with 4-bit quantization
        bnb_4bit_quant_type="nf4",       # Quantization data type: NF4
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation, ensuring speed and stability
        bnb_4bit_use_double_quant=True,  # Use double quantization to further save memory
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=bnb_config)
    
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
    train_dataset = MSDialogSFTDataset(train_data_path, tokenizer)
    val_dataset = MSDialogSFTDataset(val_data_path, tokenizer)
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,
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
        load_best_model_at_end=True,  # Required for early stopping
        metric_for_best_model="eval_loss",  # Metric for early stopping
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
    
    print(f"MSDialog {args.domain} domain supervised fine-tuning completed! Model saved at: {output_dir}")


if __name__ == "__main__":
    main()