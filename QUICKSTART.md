# Quick Start Guide

Get the Gitcoin Deep Funding Optimizer running in 3 simple steps.

## Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum

## Installation

```bash
pip install numpy pandas scipy jupyter
```

## Execution

### Option 1: Python Script (Fastest)

```bash
python run_all_tasks.py
```

### Option 2: Jupyter Notebook (Interactive)

```bash
python -m notebook gitcoin_deep_funding_optimizer.ipynb
```

Then in the browser: **Kernel → Restart & Run All**

## Output

Check the `result/` directory for three CSV files:
- `submission_task1.csv` (98 rows)
- `submission_task2.csv` (98 rows)  
- `submission_task3.csv` (3,677 rows)

## Execution Time

- Task 1: ~5 seconds
- Task 2: ~5 seconds
- Task 3: ~3-5 minutes

## Need Help?

- **Quick reference**: See [README.md](README.md)
- **Technical details**: See [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Version history**: See [CHANGELOG.md](CHANGELOG.md)

## Troubleshooting

**Jupyter not found?**
```bash
python -m notebook gitcoin_deep_funding_optimizer.ipynb
```

**Import errors?**
```bash
pip install --upgrade numpy pandas scipy jupyter
```

**Memory issues?**
- Close other applications
- Ensure 8GB+ RAM available
