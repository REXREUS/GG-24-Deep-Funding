# Gitcoin Deep Funding Optimizer - Technical Usage Guide

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Details](#implementation-details)
4. [Execution Methods](#execution-methods)
5. [Configuration Reference](#configuration-reference)
6. [Validation Framework](#validation-framework)
7. [Recent Fixes and Updates](#recent-fixes-and-updates)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## System Architecture

### Overview

The Gitcoin Deep Funding Optimizer implements a three-tier architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepFundingPipeline                      │
│                    (Orchestration Layer)                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌──────────────────────┐           ┌──────────────────────────┐
│  PairwisePredictor   │           │ HuberScaleReconstructor  │
│  (Feature Layer)     │           │   (Optimization Layer)   │
└──────────────────────┘           └──────────────────────────┘
```

### Component Responsibilities

#### 1. HuberScaleReconstructor (Optimization Core)

**Purpose**: Solves the Bradley-Terry scale reconstruction problem using Huber loss.

**Key Methods**:
- `fit(r_ij)`: Optimizes latent scales from pairwise ratio matrix
- `transform()`: Converts scales to normalized weights
- `fit_transform(r_ij)`: Combined fit and transform operation

**Algorithm**: Iteratively Reweighted Least Squares (IRLS) via `scipy.optimize.least_squares`

#### 2. PairwisePredictor (Feature Engineering)

**Purpose**: Generates pairwise comparison ratios from repository features.

**Key Methods**:
- `predict(repos)`: Computes n×n ratio matrix r_ij
- `_extract_url_features(url)`: Extracts features from GitHub URLs

**Feature Extraction**:
- Organization name length
- Repository name length
- URL path depth

#### 3. DeepFundingPipeline (Orchestration)

**Purpose**: End-to-end workflow management for all three tasks.

**Key Methods**:
- `run_task(level)`: Executes a specific task (1, 2, or 3)
- `validate_output(df)`: Comprehensive output validation
- `_export_csv(df, filename)`: CSV export with proper formatting
- `_process_parent_group(group)`: Per-parent-group optimization
- `_load_input(level)`: Task-specific data loading

---

## Mathematical Foundation

### Bradley-Terry Model

The Bradley-Terry model represents the probability that item `i` is preferred over item `j`:

```
P(i > j) = exp(x_i) / (exp(x_i) + exp(x_j))
         = 1 / (1 + exp(x_j - x_i))
```

Where:
- `x_i`, `x_j` are latent "strength" parameters
- `exp(x_i)` represents the scale or weight of item `i`

### Pairwise Ratio Formulation

Given observed ratios `r_ij ≈ exp(x_i) / exp(x_j)`, we have:

```
log(r_ij) ≈ x_i - x_j
```

This transforms the problem into a linear system in log-space.

### Huber Loss Function

The Huber loss provides robustness against outliers:

```
L_δ(r) = { ½r²           if |r| ≤ δ
         { δ(|r| - ½δ)   if |r| > δ
```

**Properties**:
- Quadratic (L2) for small residuals: smooth, efficient optimization
- Linear (L1) for large residuals: robust to outliers
- Transition point δ = 1.0 (configurable)

### Optimization Problem

We solve:

```
minimize: Σ L_δ(x_i - x_j - d_ij)
subject to: Σ exp(x_i) = 1  (normalization constraint)
```

Where `d_ij = log(r_ij)` are the log-ratios.

### Log-Sum-Exp Trick

To avoid numerical overflow/underflow, we use:

```
log(Σ exp(x_i)) = x_max + log(Σ exp(x_i - x_max))
```

Then normalize:

```
w_i = exp(x_i - log_sum_exp(x))
```

This ensures:
1. Numerical stability (no overflow)
2. Exact normalization (Σw_i = 1.0)
3. Preservation of relative magnitudes

---

## Implementation Details

### Notebook Structure (5 Cells)

#### Cell 1: Setup & Configuration

```python
# Imports: numpy, pandas, scipy, logging, etc.
# Random seed: 42 (reproducibility)
# CONFIG dictionary: hyperparameters
# Output directory creation: result/
```

#### Cell 2: HuberScaleReconstructor Class

```python
class HuberScaleReconstructor:
    def __init__(self, delta=1.0, max_iterations=1000, tolerance=1e-8)
    def fit(self, r_ij)
    def transform(self)
    def fit_transform(self, r_ij)
    def get_metrics(self)
```

#### Cell 3: PairwisePredictor Class

```python
class PairwisePredictor:
    def __init__(self, epsilon=1e-10)
    def _extract_url_features(self, url)
    def predict(self, repos)
```

#### Cell 4: DeepFundingPipeline Class

```python
class DeepFundingPipeline:
    def __init__(self, predictor, optimizer, config)
    def _load_input(self, level)
    def _process_parent_group(self, group)
    def _export_csv(self, df, filename)
    def validate_output(self, df, input_df=None)
    def run_task(self, level)
```

#### Cell 5: Execution Loop

```python
# Initialize components
# Loop through tasks 1, 2, 3
# Execute, validate, export
# Log summary statistics
```

### Data Flow

```
Input CSV
    ↓
Load & Parse (pandas)
    ↓
Group by Parent (groupby)
    ↓
For each parent group:
    ↓
    Extract Features (PairwisePredictor)
    ↓
    Compute Ratio Matrix r_ij
    ↓
    Optimize Scales (HuberScaleReconstructor)
    ↓
    Normalize to Weights
    ↓
Concatenate Results
    ↓
Validate Output
    ↓
Export CSV
```

---

## Execution Methods

### Method 1: Standalone Python Script (Recommended)

**Advantages**:
- No Jupyter dependency
- Runs in background
- Easier to automate
- Better for production

**Usage**:

```bash
python run_all_tasks.py
```

**Output**:
```
2026-03-13 02:16:19,827 - INFO - Setup complete
2026-03-13 02:16:19,830 - INFO - EXECUTION LOOP: Processing 3 Tasks
...
2026-03-13 02:16:24,917 - INFO - TASK 1 SUMMARY
2026-03-13 02:16:24,930 - INFO - Repositories processed: 98
2026-03-13 02:16:24,932 - INFO - Execution time: 4.80 seconds
2026-03-13 02:16:24,934 - INFO - Status: ✓ SUCCESS
...
```

### Method 2: Jupyter Notebook (Interactive)

**Advantages**:
- Interactive exploration
- Cell-by-cell execution
- Inline visualization
- Better for development

**Usage**:

```bash
# Start Jupyter
python -m notebook gitcoin_deep_funding_optimizer.ipynb

# In browser:
# 1. Kernel → Restart & Run All
# 2. Wait for completion
# 3. Check result/ directory
```

### Method 3: Programmatic Execution

**For integration into larger workflows**:

```python
from run_all_tasks import DeepFundingPipeline, PairwisePredictor, HuberScaleReconstructor, CONFIG

# Initialize components
predictor = PairwisePredictor(epsilon=CONFIG['epsilon'])
optimizer = HuberScaleReconstructor(
    delta=CONFIG['huber_delta'],
    max_iterations=CONFIG['max_iterations'],
    tolerance=CONFIG['tolerance']
)
pipeline = DeepFundingPipeline(predictor, optimizer, CONFIG)

# Run specific task
output_df = pipeline.run_task(level=1)

# Validate
is_valid = pipeline.validate_output(output_df)

# Export
if is_valid:
    pipeline._export_csv(output_df, 'submission_task1.csv')
```

---

## Configuration Reference

### CONFIG Dictionary

```python
CONFIG = {
    # Optimization parameters
    'huber_delta': 1.0,              # Huber loss transition point
    'max_iterations': 1000,          # Max IRLS iterations
    'tolerance': 1e-8,               # Convergence tolerance
    
    # Validation parameters
    'normalization_tolerance': 1e-6, # Weight sum tolerance
    'epsilon': 1e-10,                # Numerical stability constant
    
    # Reproducibility
    'random_seed': 42,               # Fixed random seed
    
    # Input paths
    'task1_input': 'data/level 1/repos_to_predict.csv',
    'task2_repos': 'data/level 2/repos_to_predict.csv',
    'task2_originality': 'data/level 2/originality-predictions.csv',
    'task3_input': 'data/level 3/pairs_to_predict.csv',
    
    # Output path
    'output_dir': 'result'
}
```

### Parameter Tuning Guidelines

#### huber_delta

- **Default**: 1.0
- **Range**: (0, ∞)
- **Effect**: 
  - Smaller δ → More robust to outliers (more L1-like)
  - Larger δ → Less robust, faster convergence (more L2-like)
- **Recommendation**: Keep at 1.0 unless data has extreme outliers

#### max_iterations

- **Default**: 1000
- **Range**: [100, 10000]
- **Effect**: Maximum optimization iterations
- **Recommendation**: 1000 is sufficient for convergence in most cases

#### tolerance

- **Default**: 1e-8
- **Range**: [1e-10, 1e-6]
- **Effect**: Convergence threshold for optimization
- **Recommendation**: 1e-8 provides good balance of accuracy and speed

#### normalization_tolerance

- **Default**: 1e-6
- **Range**: [1e-8, 1e-4]
- **Effect**: Acceptable deviation from sum(weights) = 1.0
- **Recommendation**: 1e-6 is appropriate for floating-point precision

---

## Validation Framework

### Validation Checks

The `validate_output()` method performs 8 comprehensive checks:

#### 1. Non-Empty DataFrame

```python
if df is None or len(df) == 0:
    return False
```

#### 2. Required Columns

```python
required_cols = ['repo', 'parent', 'weight']
if not all(col in df.columns for col in required_cols):
    return False
```

#### 3. Numeric Weight Column

```python
if not pd.api.types.is_numeric_dtype(df['weight']):
    return False
```

#### 4. Weight Range Validation

```python
# Valid range: (0.0, 1.0] (exclusive of 0, inclusive of 1)
invalid_weights = df[(df['weight'] <= 0.0) | (df['weight'] > 1.0)]
if len(invalid_weights) > 0:
    return False
```

**Note**: Weight = 1.0 is valid for single-repo parent groups.

#### 5. Normalization Constraint

```python
for parent, group in df.groupby('parent'):
    weight_sum = group['weight'].sum()
    if abs(weight_sum - 1.0) >= tolerance:
        return False
```

#### 6. No Duplicate Pairs

```python
duplicates = df.duplicated(subset=['repo', 'parent'], keep=False)
if duplicates.any():
    return False
```

#### 7. Completeness Check

```python
if input_df is not None:
    input_repos = set(input_df['repo'].unique())
    output_repos = set(df['repo'].unique())
    missing_repos = input_repos - output_repos
    if len(missing_repos) > 0:
        return False
```

#### 8. Weight Precision

```python
# Ensures >= 6 decimal places
weight_precision_check = df['weight'].apply(
    lambda w: len(str(w).split('.')[-1]) >= 6 if '.' in str(w) else False
)
```

---

## Recent Fixes and Updates

### Version 1.1 - Weight Validation Fix

**Date**: March 13, 2026

#### Problem Statement

Tasks 2 and 3 were failing validation with the error:

```
ERROR - Validation failed: 62 weights outside range (0.0, 1.0)
ERROR - Invalid weights:
                                    repo        parent  weight
0  https://github.com/0xMiden/miden-vm       0xMiden     1.0
1  https://github.com/Certora/CertoraProver  Certora     1.0
...
```

#### Root Cause Analysis

The validation logic was incorrectly rejecting weights equal to 1.0:

```python
# Incorrect implementation
invalid_weights = df[(df['weight'] <= 0.0) | (df['weight'] >= 1.0)]
#                                                           ^^
#                                                    This was wrong
```

**Why this is incorrect**:

For parent groups containing only one repository, the mathematical requirement is:

```
sum(weights) = 1.0
```

With only one repo, this means:

```
weight_single_repo = 1.0
```

This is not only valid but **required** by the normalization constraint.

#### Solution Implementation

Updated the validation to allow `weight = 1.0`:

```python
# Correct implementation
invalid_weights = df[(df['weight'] <= 0.0) | (df['weight'] > 1.0)]
#                                                           ^
#                                                    Now correct
```

**Valid weight range**: `(0.0, 1.0]`
- Exclusive of 0: weights must be positive
- Inclusive of 1: single-repo groups can have weight = 1.0

#### Impact Assessment

| Task | Before Fix | After Fix | Single-Repo Groups |
|------|------------|-----------|-------------------|
| Task 1 | ✓ Pass | ✓ Pass | 0 (all 98 in one group) |
| Task 2 | ✗ Fail | ✓ Pass | 62 out of 74 groups |
| Task 3 | ✗ Fail | ✓ Pass | 1,344 out of 1,953 groups |

#### Files Modified

1. **gitcoin_deep_funding_optimizer.ipynb**
   - Cell 4: `DeepFundingPipeline.validate_output()` method
   - Line 280: Weight range validation logic

2. **run_all_tasks.py**
   - Line 218: `DeepFundingPipeline.validate_output()` method
   - Same validation logic update

#### Verification

All three tasks now pass validation:

```
Task 1: ✓ All validations passed (98 rows, 1 parent group)
Task 2: ✓ All validations passed (98 rows, 74 parent groups)
Task 3: ✓ All validations passed (3,677 rows, 1,953 parent groups)
```

---

## Performance Optimization

### Memory Management

#### Parent Group Isolation

```python
for parent, group in df.groupby('parent'):
    result_df = self._process_parent_group(group)
    results.append(result_df)
    
    # Explicit cleanup
    del group
    gc.collect()
```

**Benefits**:
- Processes one parent group at a time
- Releases memory immediately after processing
- Prevents memory accumulation for Task 3 (1,953 groups)

#### Vectorized Operations

```python
# Efficient: vectorized numpy operations
r_ij = np.ones((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            r_ij[i, j] = (score_i + epsilon) / (score_j + epsilon)
```

### Computational Complexity

| Task | Repos | Groups | Complexity | Time |
|------|-------|--------|------------|------|
| Task 1 | 98 | 1 | O(98²) | ~5s |
| Task 2 | 98 | 74 | O(74 × avg²) | ~5s |
| Task 3 | 3,677 | 1,953 | O(1,953 × avg²) | ~3-5min |

**Note**: Most Task 3 groups have 1-2 repos, so average group size is small.

### Optimization Tips

1. **Reduce max_iterations** if convergence is fast:
   ```python
   CONFIG['max_iterations'] = 500  # Default: 1000
   ```

2. **Increase tolerance** for faster (less precise) results:
   ```python
   CONFIG['tolerance'] = 1e-6  # Default: 1e-8
   ```

3. **Use run_all_tasks.py** instead of Jupyter for better performance

---

## Troubleshooting

### Common Issues

#### Issue 1: Jupyter Command Not Found

**Symptom**:
```
'jupyter' is not recognized as an internal or external command
```

**Solution**:
```bash
# Use Python module syntax
python -m notebook gitcoin_deep_funding_optimizer.ipynb
```

#### Issue 2: Memory Error (Task 3)

**Symptom**:
```
MemoryError: Unable to allocate array
```

**Solutions**:
1. Close other applications
2. Ensure 8GB+ RAM available
3. Run on a machine with more memory
4. Process fewer groups at a time (modify code)

#### Issue 3: Convergence Failure

**Symptom**:
```
WARNING - Optimization did not converge for parent group X
```

**Solutions**:
1. Increase `max_iterations`:
   ```python
   CONFIG['max_iterations'] = 2000
   ```
2. Adjust `huber_delta`:
   ```python
   CONFIG['huber_delta'] = 0.5  # More robust
   ```
3. Check input data for anomalies

#### Issue 4: Import Errors

**Symptom**:
```
ModuleNotFoundError: No module named 'scipy'
```

**Solution**:
```bash
pip install --upgrade numpy pandas scipy jupyter
```

#### Issue 5: Validation Failures

**Symptom**:
```
ERROR - Validation failed: Parent "X" weight sum = 0.999999
```

**Solution**:
This is likely a floating-point precision issue. Adjust tolerance:
```python
CONFIG['normalization_tolerance'] = 1e-5  # More lenient
```

---

## API Reference

### HuberScaleReconstructor

#### Constructor

```python
HuberScaleReconstructor(delta=1.0, max_iterations=1000, tolerance=1e-8)
```

**Parameters**:
- `delta` (float): Huber loss transition point
- `max_iterations` (int): Maximum optimization iterations
- `tolerance` (float): Convergence tolerance

#### Methods

##### fit(r_ij)

Fits the model to pairwise ratio matrix.

**Parameters**:
- `r_ij` (np.ndarray): n×n matrix of pairwise ratios

**Returns**:
- `self`: For method chaining

**Side Effects**:
- Sets `self.x_values`: Optimized latent scales
- Sets `self.convergence_status`: Boolean convergence flag
- Sets `self.n_iterations`: Number of iterations used
- Sets `self.final_loss`: Final loss value

##### transform()

Transforms latent scales to normalized weights.

**Returns**:
- `weights` (np.ndarray): Normalized weights summing to 1.0

**Requires**: Must call `fit()` first

##### fit_transform(r_ij)

Combined fit and transform operation.

**Parameters**:
- `r_ij` (np.ndarray): n×n matrix of pairwise ratios

**Returns**:
- `weights` (np.ndarray): Normalized weights summing to 1.0

##### get_metrics()

Returns optimization metrics.

**Returns**:
- `dict`: Dictionary with keys:
  - `convergence` (bool): Whether optimization converged
  - `iterations` (int): Number of iterations used
  - `final_loss` (float): Final loss value

---

### PairwisePredictor

#### Constructor

```python
PairwisePredictor(epsilon=1e-10)
```

**Parameters**:
- `epsilon` (float): Numerical stability constant

#### Methods

##### predict(repos)

Computes pairwise ratio matrix from repository DataFrame.

**Parameters**:
- `repos` (pd.DataFrame): DataFrame with 'repo' column containing GitHub URLs

**Returns**:
- `r_ij` (np.ndarray): n×n matrix of pairwise ratios

**Algorithm**:
```python
score_i = org_name_length + repo_name_length
r_ij[i, j] = (score_i + epsilon) / (score_j + epsilon)
```

##### _extract_url_features(url)

Extracts features from GitHub URL.

**Parameters**:
- `url` (str): GitHub repository URL

**Returns**:
- `dict`: Dictionary with keys:
  - `org_name` (str): Organization name
  - `repo_name` (str): Repository name
  - `org_name_length` (int): Length of org name
  - `repo_name_length` (int): Length of repo name
  - `path_depth` (int): Number of '/' in URL

---

### DeepFundingPipeline

#### Constructor

```python
DeepFundingPipeline(predictor, optimizer, config)
```

**Parameters**:
- `predictor` (PairwisePredictor): Feature extraction component
- `optimizer` (HuberScaleReconstructor): Optimization component
- `config` (dict): Configuration dictionary

#### Methods

##### run_task(level)

Executes a complete task (1, 2, or 3).

**Parameters**:
- `level` (int): Task level (1, 2, or 3)

**Returns**:
- `output_df` (pd.DataFrame): Results with columns [repo, parent, weight]

**Process**:
1. Load input data
2. Group by parent
3. Process each group
4. Concatenate results
5. Return DataFrame

##### validate_output(df, input_df=None)

Validates output DataFrame against all requirements.

**Parameters**:
- `df` (pd.DataFrame): Output DataFrame to validate
- `input_df` (pd.DataFrame, optional): Input DataFrame for completeness check

**Returns**:
- `bool`: True if all validations pass, False otherwise

**Validation Checks**: See [Validation Framework](#validation-framework)

##### _export_csv(df, filename)

Exports DataFrame to CSV with proper formatting.

**Parameters**:
- `df` (pd.DataFrame): DataFrame to export
- `filename` (str): Output filename (e.g., 'submission_task1.csv')

**Output Format**:
- Columns: repo, parent, weight
- Weight precision: 6 decimal places minimum
- No index column
- No trailing whitespace

##### _process_parent_group(group)

Processes a single parent group.

**Parameters**:
- `group` (pd.DataFrame): Subset of repos with same parent

**Returns**:
- `result_df` (pd.DataFrame): Results for this group

**Special Cases**:
- Empty group: Returns empty DataFrame
- Single repo: Returns weight = 1.0 (no optimization needed)
- Multiple repos: Runs full optimization pipeline

##### _load_input(level)

Loads input data for specified task level.

**Parameters**:
- `level` (int): Task level (1, 2, or 3)

**Returns**:
- `df` (pd.DataFrame): Input DataFrame with columns [repo, parent]

**Task-Specific Logic**:
- Task 1: Loads repos_to_predict.csv, parent = 'ethereum'
- Task 2: Loads repos_to_predict.csv, extracts org as parent
- Task 3: Loads pairs_to_predict.csv, renames 'dependency' to 'parent'

---

## Appendix

### CSV Output Format

All output CSV files follow this format:

```csv
repo,parent,weight
https://github.com/org1/repo1,parent1,0.123456
https://github.com/org2/repo2,parent1,0.876544
https://github.com/org3/repo3,parent2,1.000000
```

**Requirements**:
- Header: `repo,parent,weight`
- Delimiter: comma (`,`)
- Weight precision: ≥6 decimal places
- No trailing whitespace
- No index column

### Logging Format

All log messages follow this format:

```
YYYY-MM-DD HH:MM:SS,mmm - LEVEL - Message
```

**Example**:
```
2026-03-13 02:16:19,827 - INFO - Setup complete
2026-03-13 02:16:24,917 - INFO - TASK 1 SUMMARY
2026-03-13 02:16:24,930 - INFO - Repositories processed: 98
```

**Log Levels**:
- `INFO`: Normal operation messages
- `WARNING`: Non-critical issues (e.g., re-normalization)
- `ERROR`: Validation failures or processing errors
- `CRITICAL`: Severe failures (>50% parent groups failed)

### File Encoding

All files use UTF-8 encoding:
- Input CSV files: UTF-8
- Output CSV files: UTF-8
- Python source files: UTF-8
- Markdown documentation: UTF-8

---

## Conclusion

This guide provides comprehensive technical documentation for the Gitcoin Deep Funding Optimizer. For additional support or questions, please refer to the inline code comments in the Jupyter notebook or the main README.md file.

**Version**: 1.1  
**Last Updated**: March 13, 2026  
**Author**: Gitcoin GG-24 Deep Funding Team
