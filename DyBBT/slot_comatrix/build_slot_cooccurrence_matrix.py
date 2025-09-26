#!/usr/bin/env python3
"""
Build Slot Co-occurrence Matrix M
According to DyBBT algorithm requirements, extract slot co-occurrence information from MultiWOZ and MSDialog datasets and build normalized co-occurrence matrix
"""

import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pickle

# Add convlab to Python path
import sys
sys.path.append('../../..')

from convlab.util.unified_datasets_util import load_dataset


def extract_all_slots_from_dialogue(dialogue: Dict) -> Set[str]:
    """Extract all unique slot names from dialogue"""
    all_slots = set()
    
    for turn in dialogue['turns']:
        if 'state' in turn:
            state = turn['state']
            for domain, slots in state.items():
                for slot in slots.keys():
                    # Build complete slot name: domain-slot
                    full_slot_name = f"{domain}-{slot}"
                    all_slots.add(full_slot_name)
    
    return all_slots


def extract_cooccurring_slots_from_turn(turn: Dict) -> Set[str]:
    """Extract co-occurring slot set from a single dialogue turn"""
    cooccurring_slots = set()
    
    if 'state' in turn:
        state = turn['state']
        for domain, slots in state.items():
            for slot, value in slots.items():
                # Only consider slots with values (non-empty strings)
                if value and value != '':
                    full_slot_name = f"{domain}-{slot}"
                    cooccurring_slots.add(full_slot_name)
    
    return cooccurring_slots


def build_slot_cooccurrence_matrix(dataset: Dict, data_split: str = 'train', dataset_type: str = 'multiwoz') -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Build Slot Co-occurrence Matrix M
    
    Args:
        dataset: Loaded dataset (MultiWOZ or MSDialog)
        data_split: Data split to use, default 'train'
        dataset_type: Type of dataset ('multiwoz' or 'msdialog')
    
    Returns:
        Normalized slot co-occurrence probability matrix and slot list
    """
    # First collect all unique slots
    all_slots = set()
    for dialogue in dataset[data_split]:
        all_slots.update(extract_all_slots_from_dialogue(dialogue))
    
    # Convert to sorted slot list for consistent matrix
    slot_list = sorted(list(all_slots))
    slot_to_index = {slot: idx for idx, slot in enumerate(slot_list)}
    num_slots = len(slot_list)
    
    # Initialize co-occurrence count matrix
    cooccurrence_counts = np.zeros((num_slots, num_slots), dtype=int)
    
    # Count co-occurrence frequencies
    total_turns = 0
    for dialogue in dataset[data_split]:
        for turn in dialogue['turns']:
            cooccurring_slots = extract_cooccurring_slots_from_turn(turn)
            if cooccurring_slots:
                total_turns += 1
                slot_indices = [slot_to_index[slot] for slot in cooccurring_slots 
                              if slot in slot_to_index]
                
                # Update co-occurrence matrix
                for i in slot_indices:
                    for j in slot_indices:
                        cooccurrence_counts[i][j] += 1
    
    print(f"Processed {total_turns} dialogue turns")
    print(f"Found {num_slots} unique slots")
    
    # Normalization: convert co-occurrence frequency to probability
    # Avoid division by zero
    cooccurrence_matrix = np.zeros((num_slots, num_slots))
    for i in range(num_slots):
        for j in range(num_slots):
            if cooccurrence_counts[i][i] > 0:  # Use diagonal elements (single slot occurrence count) for normalization
                cooccurrence_matrix[i][j] = cooccurrence_counts[i][j] / cooccurrence_counts[i][i]
    
    # Convert to dictionary format for easy use
    cooccurrence_dict = {}
    for i, slot_i in enumerate(slot_list):
        cooccurrence_dict[slot_i] = {}
        for j, slot_j in enumerate(slot_list):
            cooccurrence_dict[slot_i][slot_j] = float(cooccurrence_matrix[i][j])
    
    return cooccurrence_dict, slot_list


def save_cooccurrence_matrix(matrix: Dict, slot_list: List[str], output_dir: str = '.', dataset_type: str = 'multiwoz'):
    """Save slot co-occurrence matrix to file"""
    # Save as JSON format
    json_output = {
        'slot_list': slot_list,
        'cooccurrence_matrix': matrix,
        'dataset_type': dataset_type
    }
    
    # Create dataset-specific filenames
    json_filename = f'{dataset_type}_slot_cooccurrence_matrix.json'
    pickle_filename = f'{dataset_type}_slot_cooccurrence_matrix.pkl'
    
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    # Save as pickle format (for fast loading)
    pickle_path = os.path.join(output_dir, pickle_filename)
    with open(pickle_path, 'wb') as f:
        pickle.dump(json_output, f)
    
    print(f"Slot co-occurrence matrix saved to:")
    print(f"  - JSON format: {json_path}")
    print(f"  - Pickle format: {pickle_path}")


def load_cooccurrence_matrix(matrix_path: str) -> Dict:
    """Load slot co-occurrence matrix from file"""
    if matrix_path.endswith('.json'):
        with open(matrix_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif matrix_path.endswith('.pkl'):
        with open(matrix_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file format, please use .json or .pkl files")


def build_multiwoz_matrix():
    """Build slot co-occurrence matrix for MultiWOZ dataset"""
    print("Loading MultiWOZ 2.1 dataset...")
    
    # Load dataset
    dataset = load_dataset('multiwoz21')
    
    print("Building slot co-occurrence matrix for MultiWOZ...")
    
    # Build slot co-occurrence matrix
    cooccurrence_matrix, slot_list = build_slot_cooccurrence_matrix(dataset, 'train', 'multiwoz')
    
    # Save matrix
    output_dir = '..'
    save_cooccurrence_matrix(cooccurrence_matrix, slot_list, output_dir, 'multiwoz')
    
    # Print statistics
    print("\nMultiWOZ slot co-occurrence matrix built successfully!")
    print(f"Matrix dimensions: {len(slot_list)} x {len(slot_list)}")
    
    # Show first 10 slots
    print("\nFirst 10 slots:")
    for slot in slot_list[:10]:
        print(f"  - {slot}")
    
    # Show high co-occurrence probability slot pairs
    print("\nHigh co-occurrence probability slot pairs examples:")
    high_cooccurrence_pairs = []
    for slot_i in slot_list[:5]:  # Only look at first 5 slots
        for slot_j in slot_list:
            prob = cooccurrence_matrix[slot_i][slot_j]
            if prob > 0.5 and slot_i != slot_j:
                high_cooccurrence_pairs.append((slot_i, slot_j, prob))
    
    # Sort by probability and show top 10
    high_cooccurrence_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (slot_i, slot_j, prob) in enumerate(high_cooccurrence_pairs[:10]):
        print(f"  {i+1}. {slot_i} ↔ {slot_j}: {prob:.3f}")
    
    return cooccurrence_matrix, slot_list


def build_msdialog_matrix():
    """Build slot co-occurrence matrix for MSDialog dataset"""
    print("\nLoading MSDialog dataset...")
    
    try:
        # Load MSDialog dataset
        dataset = load_dataset('msdialog')
        
        print("Building slot co-occurrence matrix for MSDialog...")
        
        # Build slot co-occurrence matrix
        cooccurrence_matrix, slot_list = build_slot_cooccurrence_matrix(dataset, 'train', 'msdialog')
        
        # Save matrix
        output_dir = '..'
        save_cooccurrence_matrix(cooccurrence_matrix, slot_list, output_dir, 'msdialog')
        
        # Print statistics
        print("\nMSDialog slot co-occurrence matrix built successfully!")
        print(f"Matrix dimensions: {len(slot_list)} x {len(slot_list)}")
        
        # Show first 10 slots
        print("\nFirst 10 slots:")
        for slot in slot_list[:10]:
            print(f"  - {slot}")
        
        # Show high co-occurrence probability slot pairs
        print("\nHigh co-occurrence probability slot pairs examples:")
        high_cooccurrence_pairs = []
        for slot_i in slot_list[:5]:  # Only look at first 5 slots
            for slot_j in slot_list:
                prob = cooccurrence_matrix[slot_i][slot_j]
                if prob > 0.5 and slot_i != slot_j:
                    high_cooccurrence_pairs.append((slot_i, slot_j, prob))
        
        # Sort by probability and show top 10
        high_cooccurrence_pairs.sort(key=lambda x: x[2], reverse=True)
        for i, (slot_i, slot_j, prob) in enumerate(high_cooccurrence_pairs[:10]):
            print(f"  {i+1}. {slot_i} ↔ {slot_j}: {prob:.3f}")
        
        return cooccurrence_matrix, slot_list
        
    except Exception as e:
        print(f"Warning: Could not load MSDialog dataset: {e}")
        print("MSDialog dataset might not be available in the current setup.")
        return None, None


def main():
    """Main function: Build and save slot co-occurrence matrices for both datasets"""
    print("=== Slot Co-occurrence Matrix Builder ===")
    
    # Build MultiWOZ matrix
    multiwoz_matrix, multiwoz_slots = build_multiwoz_matrix()
    
    # Build MSDialog matrix
    msdialog_matrix, msdialog_slots = build_msdialog_matrix()
    
    print("\n=== All matrices built successfully! ===")
    
    # Summary
    if multiwoz_slots:
        print(f"MultiWOZ: {len(multiwoz_slots)} unique slots")
    if msdialog_slots:
        print(f"MSDialog: {len(msdialog_slots)} unique slots")


if __name__ == "__main__":
    main()