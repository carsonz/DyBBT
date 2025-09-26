import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOTrainer:
    """
    Custom PPO trainer for LoRA fine-tuning of LLM
    """
    def __init__(self, model, tokenizer, config, policy=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.policy = policy  # Add reference to policy object
        
        # PPO hyperparameters
        self.learning_rate = config.get('learning_rate', 1e-5)
        self.gamma = config.get('gamma', 0.99)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.1)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.batch_size = config.get('batch_size', 16)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        
        # Gradient accumulation parameters
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.current_accumulation_step = 0
        
        # Initialize optimizer, only optimize LoRA parameters and value head parameters
        # Get all parameters that require gradients, excluding value_head parameters
        lora_params = [p for n, p in self.model.named_parameters() 
                      if p.requires_grad and not n.startswith('value_head')]
        value_head_params = list(self.model.value_head.parameters())
        
        self.optimizer = optim.AdamW([
            {'params': lora_params, 'lr': self.learning_rate},
            {'params': value_head_params, 'lr': self.learning_rate}
        ])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.get('cosine_t_max', 1000), 
            eta_min=config.get('cosine_eta_min', 1e-7)
        )
        
        # Store trajectory data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.dones = []

        self.model.print_trainable_parameters()
        
    def collect_data(self, queries, responses, rewards, actions, log_probs, values, entropies):
        """
        Collect training data
        """
        # Concatenate query and response into complete sequences
        for query, response, reward, action, log_prob, value, entropy in zip(queries, responses, rewards, actions, log_probs, values, entropies):
            # Ensure query and response are 1D tensors
            query_seq = query.squeeze(0) if query.dim() > 1 else query
            response_seq = response.squeeze(0) if response.dim() > 1 else response
            
            # Concatenate query and response
            full_sequence = torch.cat([query_seq, response_seq], dim=0)
            
            # Store data
            self.states.append(full_sequence)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.entropies.append(entropy)
            self.rewards.append(reward)
            self.dones.append(False)  # Assume trajectory not ended, will be updated during training
            
    def clear_data(self):
        """
        Clear stored data, but keep recent data for advantage calculation
        """
        # Keep some recent data for advantage function calculation
        keep_size = min(10, len(self.states))
        
        if len(self.states) > keep_size:
            self.states = self.states[-keep_size:]
            self.actions = self.actions[-keep_size:]
            self.rewards = self.rewards[-keep_size:]
            self.log_probs = self.log_probs[-keep_size:]
            self.values = self.values[-keep_size:]
            self.entropies = self.entropies[-keep_size:]
            self.dones = self.dones[-keep_size:]
        else:
            # If data volume is small, clear completely
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.log_probs.clear()
            self.values.clear()
            self.entropies.clear()
            self.dones.clear()
        
    def reset_gradient_accumulation(self):
        """
        Reset gradient accumulation counter
        """
        self.current_accumulation_step = 0
        self.optimizer.zero_grad()
    
    def compute_advantages(self, rewards, values, dones, last_value):
        """
        Compute advantage function
        """
        advantages = []
        returns = []
        
        # Calculate TD residual
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = last_value
                next_done = torch.tensor(1.0, device=DEVICE, requires_grad=False)
            else:
                next_value = values[i+1]
                next_done = dones[i+1]
                
            delta = rewards[i] + self.gamma * next_value * (1 - next_done) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages.insert(0, gae)
            
            # Calculate returns
            returns.insert(0, advantages[0] + values[i])
            
        # Ensure advantages and returns are 1D tensors
        advantages_tensor = torch.stack(advantages)
        returns_tensor = torch.stack(returns)
        
        if advantages_tensor.dim() > 1:
            advantages_tensor = advantages_tensor.squeeze(-1)
        if returns_tensor.dim() > 1:
            returns_tensor = returns_tensor.squeeze(-1)
            
        return advantages_tensor, returns_tensor
    
    def update_policy(self):
        """
        Update policy
        """
        if len(self.states) == 0:
            return
            
        # Handle variable-length sequences - pad to same length
        max_len = max(state.size(0) for state in self.states)
        padded_states = []
        attention_masks = []
        
        for state in self.states:
            # Calculate padding length needed
            pad_len = max_len - state.size(0)
            # Create padded state (using pad_token_id for padding)
            padded_state = torch.cat([state, torch.full((pad_len,), self.tokenizer.pad_token_id, device=DEVICE, requires_grad=False)])
            padded_states.append(padded_state)
            # Create attention mask (1 for real tokens, 0 for padding tokens)
            attention_mask = torch.cat([torch.ones(state.size(0), device=DEVICE, requires_grad=False), torch.zeros(pad_len, device=DEVICE, requires_grad=False)])
            attention_masks.append(attention_mask)
        
        # Handle variable-length sequences - pad to same length
        # First process states (already processed)
        states = torch.stack(padded_states).to(DEVICE)
        attention_masks = torch.stack(attention_masks).to(DEVICE)
        
        # Then process actions
        max_action_len = max(action.size(0) for action in self.actions)
        padded_actions = []
        for action in self.actions:
            # Calculate padding length needed
            pad_len = max_action_len - action.size(0)
            # Create padded action (using 0 for padding)
            padded_action = torch.cat([action, torch.zeros(pad_len, dtype=action.dtype, device=DEVICE, requires_grad=False)])
            padded_actions.append(padded_action)
        
        # Process log_probs - no padding, directly handle variable-length sequences
        # Convert all log_probs to list and ensure they are on the same device
        log_probs_list = [log_prob.to(DEVICE) for log_prob in self.log_probs]
        
        # Process values - no padding, directly handle variable-length sequences
        values_list = [value.to(DEVICE) for value in self.values]
        
        # Process entropies - no padding, directly handle variable-length sequences
        entropies_list = [entropy.to(DEVICE) for entropy in self.entropies]
        
        # 将数据转换为张量
        actions = torch.stack(padded_actions).to(DEVICE)
        # 使用列表而不是堆叠的张量来处理变长序列
        old_log_probs = log_probs_list
        old_values = values_list
        old_entropies = entropies_list
        rewards = torch.tensor(self.rewards, dtype=torch.float32, requires_grad=False).to(DEVICE)
        dones = torch.tensor(self.dones, dtype=torch.float32, requires_grad=False).to(DEVICE)
        
        # 计算优势和回报
        with torch.no_grad():
            # 由于现在处理的是变长序列，我们需要为每个序列单独计算优势
            all_advantages = []
            all_returns = []
            
            for i in range(len(rewards)):
                # 为每个序列单独计算优势
                seq_rewards = rewards[i].unsqueeze(0) if rewards[i].dim() == 0 else rewards[i]
                seq_values = old_values[i]
                seq_dones = dones[i].unsqueeze(0) if dones[i].dim() == 0 else dones[i]
                
                advantages, returns = self.compute_advantages(seq_rewards, seq_values, seq_dones, 0)
                all_advantages.append(advantages)
                all_returns.append(returns)
            
            # 标准化优势（在所有序列上）
            flat_advantages = torch.cat([adv.flatten() for adv in all_advantages])
            
            # 检查并处理无效值
            if torch.isnan(flat_advantages).any() or torch.isinf(flat_advantages).any() or len(flat_advantages) <= 1:
                # 如果有NaN、inf值或数据不足，保持原始优势值但不进行标准化
                normalized_advantages = all_advantages
            else:
                advantages_mean = flat_advantages.mean()
                advantages_std = flat_advantages.std()
                
                # 如果标准差为0、NaN或inf，使用默认值
                if advantages_std == 0 or torch.isnan(advantages_std) or torch.isinf(advantages_std):
                    advantages_std = torch.tensor(1.0, device=DEVICE)
                else:
                    advantages_std = advantages_std + 1e-8
                
                # 应用标准化
                normalized_advantages = []
                for advantages in all_advantages:
                    normalized_advantages.append((advantages - advantages_mean) / advantages_std)
        
        # 多次更新策略
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # 重置优化器梯度
        self.optimizer.zero_grad()
        
        for _ in range(self.ppo_epochs):
            # 创建小批量数据 - 由于处理变长序列，我们按序列索引进行批处理
            batch_size = min(self.batch_size, len(states))
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_attention_masks = attention_masks[batch_indices]
                batch_actions = actions[batch_indices]
                
                # 对于变长序列，我们按索引获取对应的列表元素
                batch_old_log_probs = [old_log_probs[i] for i in batch_indices.tolist()]
                batch_advantages = [normalized_advantages[i] for i in batch_indices.tolist()]
                batch_returns = [all_returns[i] for i in batch_indices.tolist()]
                
                # 使用当前策略重新计算新的log概率、价值和熵
                _, new_log_probs, new_values, new_entropies = self.policy.get_action_and_value_for_ppo(
                    batch_states, batch_attention_masks, batch_actions
                )
                
                # 确保维度一致
                if new_values.dim() > 1:
                    new_values = new_values.squeeze(-1)
                
                # 由于get_action_and_value_for_ppo返回的是批处理结果（单个张量），
                # 我们需要按序列分割这些张量来匹配变长序列
                batch_size = len(batch_old_log_probs)
                new_log_probs_split = torch.split(new_log_probs, [len(log_prob) for log_prob in batch_old_log_probs])
                new_values_split = torch.split(new_values, [len(value) for value in batch_old_log_probs])
                new_entropies_split = torch.split(new_entropies, [len(entropy) for entropy in batch_old_log_probs])
                
                # 计算比率 - 对于变长序列，我们需要逐序列计算
                ratios = []
                for i in range(batch_size):
                    # 确保new_log_probs和batch_old_log_probs具有相同的长度
                    min_len = min(new_log_probs_split[i].size(0), batch_old_log_probs[i].size(0))
                    ratio = torch.exp(new_log_probs_split[i][:min_len] - batch_old_log_probs[i][:min_len])
                    ratios.append(ratio)
                
                # 计算PPO损失 - 对于变长序列，我们需要逐序列计算
                policy_losses = []
                value_losses = []
                entropy_losses = []
                
                for i in range(batch_size):
                    # 确保所有张量具有相同的长度
                    min_len = min(ratios[i].size(0), batch_advantages[i].size(0), new_values_split[i].size(0), batch_returns[i].size(0))
                    
                    # 策略损失
                    surr1 = ratios[i][:min_len] * batch_advantages[i][:min_len]
                    surr2 = torch.clamp(ratios[i][:min_len], 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages[i][:min_len]
                    policy_losses.append(-torch.min(surr1, surr2).mean())
                    
                    # 价值损失
                    value_losses.append(nn.MSELoss()(new_values_split[i][:min_len], batch_returns[i][:min_len]))
                    
                    # 熵奖励
                    entropy_losses.append(-new_entropies_split[i][:min_len].mean())
                
                # 平均所有序列的损失
                policy_loss = torch.stack(policy_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                entropy_loss = torch.stack(entropy_losses).mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                loss = loss / self.gradient_accumulation_steps  # 缩放损失以实现梯度积累
                loss.backward()
                
                # 梯度积累：只在达到积累步数时更新参数
                self.current_accumulation_step += 1
                if self.current_accumulation_step >= self.gradient_accumulation_steps:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.current_accumulation_step = 0
                
                # 累计损失用于日志
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
        
        # 记录平均损失
        num_batches = (len(states) + batch_size - 1) // batch_size
        avg_policy_loss = total_policy_loss / (self.ppo_epochs * num_batches)
        avg_value_loss = total_value_loss / (self.ppo_epochs * num_batches)
        avg_entropy = total_entropy / (self.ppo_epochs * num_batches)
        
        logging.info(f"PPO update - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}")
        
        # 清空数据
        self.clear_data()
        
    def train_step(self, queries, responses, rewards, actions, log_probs, values, entropies):
        """
        执行一步训练
        """
        # 重置梯度积累计数器
        self.reset_gradient_accumulation()
        
        # 收集数据
        self.collect_data(queries, responses, rewards, actions, log_probs, values, entropies)
        
        # 更新策略
        self.update_policy()
        
        # 记录训练信息
        if len(rewards) > 0:
            avg_reward = sum(rewards) / len(rewards)
            logging.info(f"PPO training step completed. Average reward: {avg_reward:.3f}")
        
    def save_model(self, save_dir, pattern):
        """
        保存模型
        """
        # 创建保存目录
        os.makedirs(f"{save_dir}/{pattern}", exist_ok=True)
        
        # 保存LoRA适配器
        self.model.save_pretrained(f"{save_dir}/{pattern}")
        
        # 保存价值头
        torch.save(self.model.value_head.state_dict(), f"{save_dir}/{pattern}/value_head.pt")
        
        logging.info(f"Model saved to {save_dir}/{pattern}")
        
    def print_trainable_parameters(self):
        """
        打印可训练参数信息
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )