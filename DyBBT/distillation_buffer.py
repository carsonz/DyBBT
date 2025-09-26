#!/usr/bin/env python3
"""
Knowledge Distillation Buffer for DyBBT
Stores high-quality System 2 decisions for System 1 fine-tuning
"""

import random
from typing import List, Tuple, Any
from collections import deque
import logging

class DistillationBuffer:
    """FIFO buffer to store System 2 decisions for knowledge distillation"""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize distillation buffer
        
        Args:
            max_size: Maximum number of entries in the buffer
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        logging.info(f"DistillationBuffer initialized with max size: {max_size}")
    
    def add(self, state: Any, action: List, confidence: float) -> None:
        """
        Add a high-confidence System 2 decision to the buffer
        
        Args:
            state: Current dialog state
            action: System 2 action
            confidence: Self-evaluated confidence score of System 2
        """
        # Only store decisions with high confidence (> 0.9)
        if confidence > 0.9:
            self.buffer.append((state, action, confidence))
            logging.debug(f"Added decision to distillation buffer. Buffer size: {len(self.buffer)}")
        else:
            logging.debug(f"Skipped low confidence decision (confidence: {confidence})")
    
    def sample_batch(self, batch_size: int = 4) -> List[Tuple[Any, List, float]]:
        """
        Sample a batch of decisions for fine-tuning
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            List of (state, action, confidence) tuples
        """
        if len(self.buffer) == 0:
            return []
        
        actual_batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, actual_batch_size)
    
    def size(self) -> int:
        """
        Get current buffer size
        
        Returns:
            Number of entries in buffer
        """
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear all entries from buffer"""
        self.buffer.clear()
        logging.info("Distillation buffer cleared")
    
    def is_full(self) -> bool:
        """
        Check if buffer is full
        
        Returns:
            True if buffer is full, False otherwise
        """
        return len(self.buffer) >= self.max_size
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics
        
        Returns:
            Dictionary with buffer statistics
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'max_size': self.max_size,
                'is_full': False,
                'avg_confidence': 0.0
            }
        
        confidences = [entry[2] for entry in self.buffer]
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'is_full': self.is_full(),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences)
        }