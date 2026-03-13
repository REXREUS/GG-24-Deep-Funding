# Quick Start Guide

## Running the Optimizer

### Option 1: Python Script (Fastest)

```bash
python run_all_tasks.py
```

This will execute all three tasks and generate CSV files in the `result/` directory.

### Option 2: Jupyter Notebook

```bash
# Start Jupyter
python -m notebook gitcoin_deep_funding_optimizer.ipynb

# In the notebook:
# - Click "Kernel" → "Restart & Run All"
# - Wait for completion
```

## Expected Output

After execution, you'll find three files in `result/`:

1. **submission_task1.csv** (98 rows)
   - Format: `repo,parent,weight`
   - Method: Bradley-Terry optimization
   - Time: ~5 seconds

2. **submission_task2.csv** (98 rows)
   - Format: `repo,originality`
   - Method: Direct originality score assignment
   - Time: <1 second

3. **submission_task3.csv** (3,677 rows)
   - Format: `repo,parent,weight`
   - Method: Bradley-Terry optimization
   - Time: ~3-5 minutes

## Task 2 Implementation Note

Task 2 uses a different methodology than Tasks 1 and 3:

- **Tasks 1 & 3**: Use Bradley-Terry model optimization to compute relative weights
- **Task 2**: Directly assigns pre-computed originality scores (no optimization)

This is why Task 2 has a different output format (`repo,originality` instead of `repo,parent,weight`).

## Validation

All outputs are automatically validated:

- ✓ Column format correctness
- ✓ Value range validation
- ✓ Normalization constraints (Tasks 1 & 3)
- ✓ Completeness checks

## Troubleshooting

### Jupyter not found
```bash
pip install jupyter notebook
```

### Import errors
```bash
pip install numpy pandas scipy
```

### Memory issues (Task 3)
- Ensure at least 8GB RAM available
- Close other applications

## Next Steps

After successful execution:

1. Verify output files in `result/` directory
2. Check file formats match expected structure
3. Review validation logs for any warnings
4. Proceed with submission

For detailed documentation, see:
- `README.md` - Project overview and mathematical framework
- `USAGE_GUIDE.md` - Technical details and architecture
- `PROJECT_SUMMARY.md` - High-level project summary
