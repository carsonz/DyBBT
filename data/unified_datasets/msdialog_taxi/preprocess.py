import os
import json
import pickle
import random
from typing import Dict, List, Any
import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

def load_pickle_file(file_path: str) -> Any:
    """加载pickle文件"""
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def convert_msdialog_to_unified_format(domain: str, data_dir: str) -> Dict[str, List[Dict]]:
    """将MSDialog格式转换为unified_datasets格式"""
    
    # 加载数据
    user_goals = load_pickle_file(os.path.join(data_dir, 'user_goals_first.v4.p'))
    kb_data = load_pickle_file(os.path.join(data_dir, 'taxi.kb.1k.v1.p'))
    
    # 读取slot信息
    with open(os.path.join(data_dir, 'taxi_slots.txt'), 'r', encoding='utf-8') as f:
        slots = [line.strip() for line in f.readlines() if line.strip()]
    
    # 读取dialogue acts
    with open(os.path.join(data_dir, 'dia_acts.txt'), 'r', encoding='utf-8') as f:
        dialogue_acts = [line.strip() for line in f.readlines() if line.strip()]
    
    # 构建ontology
    ontology = {
        "domains": {
            "taxi": {
                "description": "book taxi service",
                "slots": {}
            }
        }
    }
    
    # 添加slots到ontology
    for slot in slots:
        ontology["domains"]["taxi"]["slots"][slot] = {
            "description": f"{slot} information for taxi booking",
            "is_categorical": False,
            "possible_values": []
        }
    
    # 构建数据库
    database = []
    for i, entry in kb_data.items():
        database.append({
            "id": i,
            **entry
        })
    
    # 构建对话数据
    dialogues = []
    for i, goal in enumerate(user_goals):
        dialogue_id = f"msdialog_taxi_{i:04d}"
        
        # 构建goal
        goal_description = "You want to book a taxi. "
        inform_slots = goal.get('inform_slots', {})
        for slot, value in inform_slots.items():
            goal_description += f"The {slot} should be {value}. "
        
        # 构建对话回合
        turns = []
        turns.append({
            "speaker": "user",
            "utterance": goal_description.strip(),
            "utt_idx": 0,
            "dialogue_acts": {
                "categorical": [],
                "non-categorical": [],
                "binary": []
            },
            "state": {"taxi": inform_slots}
        })
        
        dialogues.append({
            "dataset": "msdialog_taxi",
            "data_split": "train",  # 所有数据都放在train中，后续可以分割
            "dialogue_id": dialogue_id,
            "original_id": str(i),
            "domains": ["taxi"],
            "goal": {
                "description": goal_description,
                "inform": {"taxi": inform_slots},
                "request": {}
            },
            "turns": turns
        })
    
    # 随机分割数据
    random.shuffle(dialogues)
    total = len(dialogues)
    train_size = int(total * 0.7)
    val_size = int(total * 0.15)
    
    train_data = dialogues[:train_size]
    val_data = dialogues[train_size:train_size + val_size]
    test_data = dialogues[train_size + val_size:]
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
        "ontology": ontology,
        "database": database
    }

def main():
    """主函数"""
    print("开始处理MSDialog Taxi数据集...")
    
    # 原始数据路径
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_dir = os.path.join(base_dir, "EIERL", "src", "deep_dialog", "data_taxi")
    
    # 转换数据
    dataset = convert_msdialog_to_unified_format("taxi", data_dir)
    
    # 保存数据
    output_dir = os.path.dirname(__file__)
    
    with open(os.path.join(output_dir, 'dialogues.json'), 'w', encoding='utf-8') as f:
        json.dump({"train": dataset["train"], "validation": dataset["validation"]}, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, 'ontology.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset["ontology"], f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, 'database.py'), 'w', encoding='utf-8') as f:
        f.write("database = " + json.dumps(dataset["database"], indent=2, ensure_ascii=False))
    
    # 创建dummy数据
    dummy_data = dataset["train"][:10] if dataset["train"] else []
    with open(os.path.join(output_dir, 'dummy_data.json'), 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=2, ensure_ascii=False)
    
    # 创建shuffled IDs
    all_ids = [d["dialogue_id"] for d in dataset["train"] + dataset["validation"]]
    random.shuffle(all_ids)
    with open(os.path.join(output_dir, 'shuffled_dial_ids.json'), 'w', encoding='utf-8') as f:
        json.dump(all_ids, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成！训练集: {len(dataset['train'])}条, 验证集: {len(dataset['validation'])}条")

if __name__ == "__main__":
    main()