# Slot Co-occurrence Matrix Construction

## Function Description

This module implements the slot co-occurrence matrix $M$ construction function in the DyBBT algorithm, used for calculating slot dependency $\rho_t$.

## File Description

- `build_slot_cooccurrence_matrix.py` - Main construction script
- `test_cooccurrence_matrix.py` - Test script
- `slot_cooccurrence_matrix.json` - Co-occurrence matrix in JSON format
- `slot_cooccurrence_matrix.pkl` - Co-occurrence matrix in Pickle format

## Usage Instructions

### 1. Build Slot Co-occurrence Matrix

```bash
cd ../../../DyBBT
PYTHONPATH=/home/zsy/workspace/gitee/DyBBT python3 slot_comatrix/build_slot_cooccurrence_matrix.py
```

### 2. Load and Use Matrix

```python
import json

# Load matrix
def load_cooccurrence_matrix(matrix_path):
    with open(matrix_path, 'r', encoding='utf-8') as f:
        return json.load(f)

matrix_data = load_cooccurrence_matrix('slot_cooccurrence_matrix.json')
slot_list = matrix_data['slot_list']
matrix = matrix_data['cooccurrence_matrix']

# Calculate slot dependency ρ_t
def calculate_slot_dependency(known_slots, unknown_slots, cooccurrence_matrix):
    """
    Calculate dependency of unknown slots on known slots
    
    Args:
        known_slots: List of known slots
        unknown_slots: List of unknown slots
        cooccurrence_matrix: Slot co-occurrence matrix
    
    Returns:
        Maximum dependency ρ_t
    """
    dependency_scores = {}
    
    for u in unknown_slots:
        if u in cooccurrence_matrix:
            total_prob = 0
            count = 0
            for f in known_slots:
                if f in cooccurrence_matrix[u]:
                    total_prob += cooccurrence_matrix[u][f]
                    count += 1
            if count > 0:
                avg_prob = total_prob / count
                dependency_scores[u] = avg_prob
    
    return max(dependency_scores.values()) if dependency_scores else 0.0

# Usage example
known_slots = ["hotel-area", "hotel-price range"]
unknown_slots = ["hotel-name", "hotel-type"]
rho_t = calculate_slot_dependency(known_slots, unknown_slots, matrix)
print(f"Slot dependency ρ_t = {rho_t:.3f}")
```

## Matrix Structure

The co-occurrence matrix contains 31 slots, using the `domain-slot` naming format:

```json
{
  "slot_list": ["attraction-area", "attraction-name", ...],
  "cooccurrence_matrix": {
    "attraction-area": {
      "attraction-area": 1.0,
      "attraction-name": 0.561,
      "attraction-type": 0.786,
      ...
    },
    ...
  }
}
```

## High Co-occurrence Probability Slot Pairs Examples

| Slot Pair | Co-occurrence Probability |
|-----------|--------------------------|
| attraction-area ↔ attraction-type | 0.786 |
| attraction-type ↔ attraction-area | 0.749 |
| attraction-name ↔ attraction-type | 0.593 |
| hotel-area ↔ hotel-type | 0.584 |
| hotel-area ↔ hotel-price range | 0.576 |
| attraction-name ↔ attraction-area | 0.561 |
| restaurant-food ↔ restaurant-price range | 0.750 |
| taxi-departure ↔ taxi-destination | 0.905 |

## Technical Details

1. **Data Source**: MultiWOZ 2.1 training set (54,668 dialogue turns)
2. **Slot Extraction**: Extract all slots with values from dialogue states
3. **Co-occurrence Statistics**: Count co-occurrences of slots in the same dialogue turn
4. **Normalization**: Normalize using individual slot occurrence counts to get co-occurrence probabilities
5. **Persistence**: Support both JSON and Pickle formats for saving

## Application in DyBBT Algorithm

The slot co-occurrence matrix is used to calculate the slot dependency $\rho_t$ in cognitive states:

$$\rho_t = \max_{u \in U} \left( \frac{1}{|F|} \sum_{f \in F} M(u, f) \right)$$

Where:
- $F$: Set of known slots
- $U$: Set of unknown slots  
- $M(u, f)$: Co-occurrence probability between slots $u$ and $f$

This metric reflects the dependency degree of unknown slots on known slots and is an important input for the DyBBT meta-controller's decision-making.