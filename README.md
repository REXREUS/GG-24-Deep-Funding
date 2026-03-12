# Gitcoin Deep Funding Optimizer

A mathematical optimization system for allocating $350,000 to Ethereum open source projects using the Bradley-Terry model with Huber loss optimization.

## Overview

This system computes relative weights for repositories based on pairwise comparisons, processing three allocation tasks with increasing complexity:

- **Task 1**: Single-parent allocation (98 repos, parent: 'ethereum')
- **Task 2**: Multi-parent allocation (98 repos, 74 parent groups)
- **Task 3**: Dependency graph allocation (3,677 pairs, 1,953 parent groups)

## Key Features

- **Mathematical Rigor**: Bradley-Terry model in log-space for numerical stability
- **Robust Optimization**: Huber loss function resistant to outliers
- **Memory Efficient**: Pandas groupby operations for parent isolation
- **Reproducible**: Fixed random seed (42) for consistent results
- **Validated**: Comprehensive checks ensuring sum(weights) = 1.0 per parent group

## Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (recommended for Task 3)

### Installation

```bash
# Install required dependencies
pip install numpy pandas scipy jupyter
```

### Running the Optimizer

**Option 1: Direct Python Script (Recommended)**

```bash
python run_all_tasks.py
```

This will execute all three tasks sequentially and generate output CSV files in the `result/` directory.

**Option 2: Jupyter Notebook**

```bash
# Start Jupyter Notebook
python -m notebook gitcoin_deep_funding_optimizer.ipynb

# In the notebook interface:
# - Select "Kernel" → "Restart & Run All"
# - Wait for all cells to complete execution
```

### Output Files

After execution, three CSV files will be generated in the `result/` directory:

- `submission_task1.csv` - Task 1 results (98 rows)
- `submission_task2.csv` - Task 2 results (98 rows)
- `submission_task3.csv` - Task 3 results (3,677 rows)

Each CSV file contains three columns: `repo`, `parent`, `weight`

## Project Structure

```
.
├── gitcoin_deep_funding_optimizer.ipynb  # Main Jupyter notebook
├── run_all_tasks.py                      # Standalone execution script
├── data/                                 # Input CSV files
│   ├── level 1/
│   │   └── repos_to_predict.csv
│   ├── level 2/
│   │   └── repos_to_predict.csv
│   └── level 3/
│       └── pairs_to_predict.csv
├── result/                               # Output CSV files (generated)
│   ├── submission_task1.csv
│   ├── submission_task2.csv
│   └── submission_task3.csv
├── README.md                             # This file
└── USAGE_GUIDE.md                        # Detailed technical documentation
```

## Mathematical Framework

### Bradley-Terry Model

The system uses the Bradley-Terry model for pairwise comparisons:

```
P(i beats j) = exp(x_i) / (exp(x_i) + exp(x_j))
```

Where `x_i` represents the latent "strength" or "quality" of repository `i`.

### Huber Loss Optimization

Optimization is performed using Huber loss (δ=1.0) via `scipy.optimize.least_squares`:

```
L(r) = { ½r²           if |r| ≤ δ
       { δ(|r| - ½δ)   if |r| > δ
```

This provides robustness against outliers while maintaining efficiency for small residuals.

### Log-Sum-Exp Normalization

Weights are normalized using the numerically stable log-sum-exp trick:

```
w_i = exp(x_i - log_sum_exp(x))
```

Where `log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))`

## Recent Fixes

### Weight Validation Fix (v1.1)

**Issue**: Task 2 and Task 3 validation was failing due to overly strict weight range checking.

**Root Cause**: The validation logic rejected weights equal to 1.0:
```python
# Old (incorrect)
invalid_weights = df[(df['weight'] <= 0.0) | (df['weight'] >= 1.0)]
```

This caused failures for parent groups with only one repository, where `weight = 1.0` is mathematically correct and required.

**Solution**: Updated validation to allow `weight = 1.0`:
```python
# New (correct)
invalid_weights = df[(df['weight'] <= 0.0) | (df['weight'] > 1.0)]
```

**Impact**:
- Task 1: No change (already passing)
- Task 2: Now passes validation (62 single-repo parent groups with weight=1.0)
- Task 3: Now passes validation (1,344 single-repo parent groups with weight=1.0)

**Valid Weight Range**: `(0.0, 1.0]` (exclusive of 0, inclusive of 1)

## Configuration

Key hyperparameters in `CONFIG` dictionary:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `huber_delta` | 1.0 | Huber loss transition point |
| `max_iterations` | 1000 | Maximum optimization iterations |
| `tolerance` | 1e-8 | Convergence tolerance |
| `normalization_tolerance` | 1e-6 | Weight sum validation tolerance |
| `epsilon` | 1e-10 | Numerical stability constant |
| `random_seed` | 42 | Reproducibility seed |

## Performance

Typical execution times on modern hardware:

- **Task 1**: ~5 seconds (98 repos, 1 parent group)
- **Task 2**: ~5 seconds (98 repos, 74 parent groups)
- **Task 3**: ~3-5 minutes (3,677 pairs, 1,953 parent groups)

## Validation

The system performs comprehensive validation on all outputs:

1. **Column Presence**: Ensures `repo`, `parent`, `weight` columns exist
2. **Weight Range**: Validates `0.0 < weight ≤ 1.0`
3. **Weight Precision**: Ensures ≥6 decimal places
4. **Normalization**: Verifies `sum(weights) = 1.0` per parent group (±1e-6)
5. **Uniqueness**: Checks for duplicate (repo, parent) pairs
6. **Completeness**: Confirms all input repos present in output

## Troubleshooting

### Jupyter Command Not Found

If `jupyter notebook` command is not recognized:

```bash
# Use Python module syntax instead
python -m notebook gitcoin_deep_funding_optimizer.ipynb
```

### Memory Issues (Task 3)

If Task 3 fails due to memory constraints:

- Close other applications
- Ensure at least 8GB RAM available
- Consider running on a machine with more memory

### Import Errors

If you encounter import errors:

```bash
# Reinstall dependencies
pip install --upgrade numpy pandas scipy jupyter
```

## Technical Documentation

For detailed technical information, including:
- Architecture design
- Algorithm implementation details
- Mathematical derivations
- API reference

Please refer to [USAGE_GUIDE.md](USAGE_GUIDE.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is part of the Gitcoin GG-24 Deep Funding competition.

## Support

For issues or questions, please refer to the comprehensive documentation in `USAGE_GUIDE.md` or review the inline code comments in the Jupyter notebook.
