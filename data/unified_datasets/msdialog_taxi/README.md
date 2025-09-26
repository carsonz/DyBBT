# MSDialog Taxi Dataset

This dataset contains taxi booking dialogues from the Microsoft Dialogue (MSDialog) corpus, converted to the unified format for ConvLab-3.

## Dataset Information

- **Source**: Microsoft Dialogue (MSDialog) corpus
- **Domain**: Taxi service booking
- **Format**: Unified ConvLab-3 format
- **Size**: Varies based on original MSDialog data

## Files

- `dialogues.json`: Contains train/validation split dialogues
- `ontology.json`: Domain ontology with slots and descriptions
- `database.py`: Knowledge base for taxi information
- `dummy_data.json`: Small sample of data for testing
- `shuffled_dial_ids.json`: Shuffled dialogue IDs for training
- `preprocess.py`: Script to regenerate the dataset

## Usage

```python
from convlab.util.unified_datasets_util import load_dataset

dataset = load_dataset('msdialog_taxi')
```

## Original Data Source

The original data comes from the EIERL project's MSDialog taxi dataset located at:
`EIERL/src/deep_dialog/data_taxi/`