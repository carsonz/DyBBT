# Dataset Preprocessing Summary

## Completed Work

### 1. MSDialog Dataset Conversion
Converted three MSDialog domain datasets from EIERL format to ConvLab-3 unified format:

- **msdialog_movie**: Movie ticket booking domain
  - Training set: 1,844 dialogues
  - Validation set: 395 dialogues  
  - Test set: 396 dialogues

- **msdialog_restaurant**: Restaurant reservation domain
  - Training set: 2,467 dialogues
  - Validation set: 528 dialogues
  - Test set: 530 dialogues

- **msdialog_taxi**: Taxi ordering domain
  - Training set: 1,980 dialogues
  - Validation set: 424 dialogues
  - Test set: None (original data does not include test set)

### 2. Data Preprocessing Script Improvements
Improved the `/home/zsy/workspace/gitee/DyBBT/DyBBT/finetune/sft/dataprepare.py` script to support:

- **MultiWOZ 2.1** dataset processing
- **MSDialog** three domain datasets processing
- Local dataset loading (avoiding network downloads)
- Automatic recognition of different data formats

### 3. Generated Processed Files
The following files were generated in the `/home/zsy/workspace/gitee/DyBBT/DyBBT/finetune/sft/` directory:

#### MultiWOZ 2.1
- `multiwoz21_train.json` - Training data
- `multiwoz21_val.json` - Validation data  
- `multiwoz21_test.json` - Test data

#### MSDialog Movie
- `msdialog_movie_train.json` - Training data
- `msdialog_movie_val.json` - Validation data
- `msdialog_movie_test.json` - Test data

#### MSDialog Restaurant  
- `msdialog_restaurant_train.json` - Training data
- `msdialog_restaurant_val.json` - Validation data
- `msdialog_restaurant_test.json` - Test data

#### MSDialog Taxi
- `msdialog_taxi_train.json` - Training data
- `msdialog_taxi_val.json` - Validation data

## Data Format Description

Each sample contains the following fields:

- `dialog_id`: Dialogue ID (format: datasetName_dialogueID)
- `domain`: List of domains
- `belief_state`: Belief state (core input)
- `system_action`: System action (training label)
- `available_actions`: Available actions under current state

## Usage Instructions

### 1. Run Preprocessing Script
```bash
cd /home/zsy/workspace/gitee/DyBBT/DyBBT/finetune/sft
python dataprepare.py
```

### 2. Load Processed Data
```python
import json

# Load training data
with open('multiwoz21_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Load validation data  
with open('multiwoz21_val.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
```

## Notes

1. The MSDialog dataset only contains user turns without system turns, so the `system_action` field is an empty list.
2. The `available_actions` field in the MSDialog dataset is also empty because there are no system actions to choose from.
3. All datasets have been converted to a unified format and can be directly used for model training.

## Next Steps

These preprocessed datasets can now be used for:

1. Supervised Fine-Tuning (SFT) training
2. Reinforcement Learning (RL) training
3. Model evaluation and testing

The data is ready for subsequent experiments in the DyBBT project.