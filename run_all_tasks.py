#!/usr/bin/env python3
"""
Gitcoin Deep Funding Optimizer - Direct Execution Script
Runs all three tasks and generates submission CSV files.
"""

import numpy as np
import pandas as pd
import logging
import random
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional
from scipy.optimize import least_squares
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CONFIG = {
    'huber_delta': 1.0,
    'max_iterations': 1000,
    'tolerance': 1e-8,
    'normalization_tolerance': 1e-6,
    'epsilon': 1e-10,
    'random_seed': RANDOM_SEED,
    'task1_input': 'data/level 1/repos_to_predict.csv',
    'task2_repos': 'data/level 2/repos_to_predict.csv',
    'task2_originality': 'data/level 2/originality-predictions.csv',
    'task3_input': 'data/level 3/pairs_to_predict.csv',
    'output_dir': 'result'
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
logger.info('Setup complete')


class HuberScaleReconstructor:
    def __init__(self, delta=1.0, max_iterations=1000, tolerance=1e-8):
        self.delta = delta
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.x_values = None
        self.convergence_status = False
        self.n_iterations = 0
        self.final_loss = 0.0
    
    def fit(self, r_ij):
        n = r_ij.shape[0]
        d_ij = np.log(r_ij + 1e-10)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        d_values = np.array([d_ij[i, j] for i, j in pairs])
        
        def residuals(x):
            return np.array([x[i] - x[j] - d_values[k] for k, (i, j) in enumerate(pairs)])
        
        x0 = np.zeros(n)
        result = least_squares(residuals, x0, loss='huber', f_scale=self.delta, max_nfev=self.max_iterations, ftol=self.tolerance)
        self.x_values = result.x
        self.convergence_status = result.success
        self.n_iterations = result.nfev
        self.final_loss = np.sum(result.fun**2)
        return self
    
    def transform(self):
        x = self.x_values
        x_max = np.max(x)
        log_sum = x_max + np.log(np.sum(np.exp(x - x_max)))
        weights = np.exp(x - log_sum)
        return weights
    
    def fit_transform(self, r_ij):
        self.fit(r_ij)
        return self.transform()
    
    def get_metrics(self):
        return {'convergence': self.convergence_status, 'iterations': self.n_iterations, 'final_loss': self.final_loss}


class PairwisePredictor:
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon
        self.feature_cache = {}
    
    def _extract_url_features(self, url):
        try:
            match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
            if match:
                org_name = match.group(1)
                repo_name = match.group(2)
            else:
                org_name = 'unknown'
                repo_name = 'unknown'
        except:
            org_name = 'unknown'
            repo_name = 'unknown'
        return {'org_name': org_name, 'repo_name': repo_name, 'org_name_length': len(org_name), 'repo_name_length': len(repo_name), 'path_depth': url.count('/')}
    
    def predict(self, repos):
        n = len(repos)
        features = [self._extract_url_features(url) for url in repos['repo'].values]
        r_ij = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    score_i = features[i]['org_name_length'] + features[i]['repo_name_length']
                    score_j = features[j]['org_name_length'] + features[j]['repo_name_length']
                    r_ij[i, j] = (score_i + self.epsilon) / (score_j + self.epsilon)
        return r_ij


class DeepFundingPipeline:
    def __init__(self, predictor, optimizer, config):
        self.predictor = predictor
        self.optimizer = optimizer
        self.config = config
        self.logger = logging.getLogger(f'{__name__}.DeepFundingPipeline')
        self.results = {}
        self.logger.info('DeepFundingPipeline initialized')
    
    def _load_input(self, level):
        self.logger.info('='*80)
        self.logger.info(f'Loading input for Task {level}')
        self.logger.info('='*80)
        if level == 1:
            df = pd.read_csv(self.config['task1_input'])
            self.logger.info(f'Loaded {len(df)} rows')
            return df
        elif level == 2:
            # Task 2: Simply use originality scores (no optimization needed)
            originality_df = pd.read_csv(self.config['task2_originality'])
            self.logger.info(f'Loaded {len(originality_df)} repos with originality scores')
            return originality_df
        elif level == 3:
            df = pd.read_csv(self.config['task3_input'])
            df = df.rename(columns={'dependency': 'parent'})
            def normalize_url(url):
                if not url.startswith('http'):
                    return f'https://github.com/{url}'
                return url
            df['repo'] = df['repo'].apply(normalize_url)
            df['parent'] = df['parent'].apply(normalize_url)
            self.logger.info(f'Loaded {len(df)} pairs')
            return df
        else:
            raise ValueError(f'Invalid level: {level}')
    
    def _process_parent_group(self, group):
        parent_name = group['parent'].iloc[0] if len(group) > 0 else 'unknown'
        n_repos = len(group)
        try:
            self.logger.info(f'Processing: {parent_name} ({n_repos} repos)')
            if n_repos == 0:
                return pd.DataFrame(columns=['repo', 'parent', 'weight'])
            if n_repos == 1:
                return pd.DataFrame({'repo': group['repo'].values, 'parent': group['parent'].values, 'weight': [1.0]})
            r_ij = self.predictor.predict(group)
            weights = self.optimizer.fit_transform(r_ij)
            weights_sum = np.sum(weights)
            if abs(weights_sum - 1.0) >= 1e-6:
                self.logger.warning(f'Re-normalizing: sum={weights_sum:.10f}')
                weights = weights / weights_sum
            result_df = pd.DataFrame({'repo': group['repo'].values, 'parent': group['parent'].values, 'weight': weights})
            self.logger.info(f'✓ {parent_name}: sum={np.sum(weights):.10f}')
            return result_df
        except Exception as e:
            self.logger.error(f'✗ Failed {parent_name}: {e}')
            return pd.DataFrame(columns=['repo', 'parent', 'weight'])
    
    def _export_csv(self, df, filename, level=None):
        self.logger.info(f'Exporting to {filename}')
        df_export = df.copy()
        
        # Task 2 has different format: repo,originality (no weight formatting needed)
        if level == 2:
            # Ensure originality has proper precision
            if 'originality' in df_export.columns:
                df_export['originality'] = df_export['originality'].apply(lambda x: f'{x:.2f}')
                output_path = Path(self.config['output_dir']) / filename
                df_export.to_csv(output_path, index=False, columns=['repo', 'originality'])
        else:
            # Tasks 1 and 3: repo,parent,weight format
            df_export['weight'] = df_export['weight'].apply(lambda x: f'{x:.6f}')
            output_path = Path(self.config['output_dir']) / filename
            df_export.to_csv(output_path, index=False, columns=['repo', 'parent', 'weight'])
        
        self.logger.info(f'✓ Exported {len(df_export)} rows to {filename}')
    
    def validate_output(self, df, input_df=None, level=None):
        self.logger.info('Starting output validation')
        validation_passed = True
        
        # Check 1: DataFrame not empty
        if df is None or len(df) == 0:
            self.logger.error('Validation failed: Empty DataFrame')
            return False
        
        # Task 2 has different validation (repo,originality format)
        if level == 2:
            # Check required columns for Task 2
            if not all(col in df.columns for col in ['repo', 'originality']):
                self.logger.error('Validation failed: Missing required columns (repo, originality)')
                return False
            
            # Check originality is numeric
            if not pd.api.types.is_numeric_dtype(df['originality']):
                self.logger.error('Validation failed: Originality column is not numeric')
                return False
            
            # Check originality range (0.0, 1.0]
            invalid_originality = df[(df['originality'] <= 0.0) | (df['originality'] > 1.0)]
            if len(invalid_originality) > 0:
                self.logger.error(f'Validation failed: {len(invalid_originality)} originality values outside range (0.0, 1.0]')
                validation_passed = False
            
            if validation_passed:
                self.logger.info('✓ All validations passed')
            else:
                self.logger.error('✗ Validation failed')
            
            return validation_passed
        
        # Tasks 1 and 3: Standard validation (repo,parent,weight format)
        # Check 2: Required columns present (Req 11.1)
        if not all(col in df.columns for col in ['repo', 'parent', 'weight']):
            self.logger.error('Validation failed: Missing required columns (repo, parent, weight)')
            return False
        
        # Check 3: Weight column is numeric
        if not pd.api.types.is_numeric_dtype(df['weight']):
            self.logger.error('Validation failed: Weight column is not numeric')
            return False
        
        # Check 4: Weight range validation (Req 11.4, 20.2)
        invalid_weights = df[(df['weight'] <= 0.0) | (df['weight'] > 1.0)]
        if len(invalid_weights) > 0:
            self.logger.error(f'Validation failed: {len(invalid_weights)} weights outside valid range (0.0 < weight <= 1.0)')
            self.logger.error(f'Invalid weights: {invalid_weights[["repo", "parent", "weight"]].head()}')
            validation_passed = False
        
        # Check 5: Normalization constraint per parent group (Req 11.8, 20.1)
        grouped = df.groupby('parent')
        tolerance = self.config.get('normalization_tolerance', 1e-6)
        for parent, group in grouped:
            weight_sum = group['weight'].sum()
            if abs(weight_sum - 1.0) >= tolerance:
                self.logger.error(f'Validation failed: Parent "{parent}" weight sum = {weight_sum:.10f} (expected 1.0 ± {tolerance})')
                validation_passed = False
        
        # Check 6: No duplicate (repo, parent) pairs (Req 11.6)
        duplicates = df.duplicated(subset=['repo', 'parent'], keep=False)
        if duplicates.any():
            n_duplicates = duplicates.sum()
            self.logger.error(f'Validation failed: {n_duplicates} duplicate (repo, parent) pairs found')
            validation_passed = False
        
        # Check 7: All input repos present in output (Req 11.7, 20.3)
        if input_df is not None:
            input_repos = set(input_df['repo'].unique())
            output_repos = set(df['repo'].unique())
            missing_repos = input_repos - output_repos
            if len(missing_repos) > 0:
                self.logger.error(f'Validation failed: {len(missing_repos)} input repos missing from output')
                validation_passed = False
        
        if validation_passed:
            self.logger.info('✓ All validations passed')
        else:
            self.logger.error('✗ Validation failed')
        
        return validation_passed
    
    def run_task(self, level):
        self.logger.info(f'Starting Task {level} execution')
        df = self._load_input(level)
        
        # Task 2 is special - no optimization needed, just return originality scores
        if level == 2:
            self.logger.info(f'Task {level} complete: {len(df)} rows (originality scores)')
            return df
        
        grouped = df.groupby('parent')
        n_parent_groups = len(grouped)
        self.logger.info(f'Found {n_parent_groups} parent groups')
        results = []
        failed_parents = []
        for parent, group in grouped:
            result_df = self._process_parent_group(group)
            if len(result_df) > 0:
                results.append(result_df)
            else:
                failed_parents.append(parent)
            del group
            gc.collect()
        if len(failed_parents) > 0:
            failure_rate = len(failed_parents) / n_parent_groups
            self.logger.warning(f'Failed parent groups: {len(failed_parents)}/{n_parent_groups} ({failure_rate*100:.1f}%)')
            if failure_rate > 0.5:
                self.logger.critical(f'CRITICAL: More than 50% of parent groups failed!')
        if len(results) == 0:
            self.logger.error('No results generated')
            return pd.DataFrame(columns=['repo', 'parent', 'weight'])
        output_df = pd.concat(results, ignore_index=True)
        self.logger.info(f'Task {level} complete: {len(output_df)} rows generated')
        return output_df


def main():
    logger.info('='*80)
    logger.info('EXECUTION LOOP: Processing 3 Tasks')
    logger.info('='*80)
    logger.info('')
    
    predictor = PairwisePredictor(epsilon=CONFIG['epsilon'])
    optimizer = HuberScaleReconstructor(delta=CONFIG['huber_delta'], max_iterations=CONFIG['max_iterations'], tolerance=CONFIG['tolerance'])
    pipeline = DeepFundingPipeline(predictor, optimizer, CONFIG)
    
    for task_level in [1, 2, 3]:
        logger.info('')
        logger.info('='*80)
        logger.info(f'TASK {task_level} EXECUTION')
        logger.info('='*80)
        
        start_time = time.time()
        output_df = pipeline.run_task(task_level)
        
        logger.info('')
        logger.info(f'Validating Task {task_level} output...')
        validation_passed = pipeline.validate_output(output_df, level=task_level)
        
        if validation_passed:
            pipeline._export_csv(output_df, f'submission_task{task_level}.csv', level=task_level)
            logger.info('')
            logger.info('='*80)
            logger.info(f'TASK {task_level} SUMMARY')
            logger.info('='*80)
            logger.info(f'Repositories processed: {len(output_df)}')
            if task_level == 2:
                logger.info(f'Originality scores: {len(output_df)} repos')
            else:
                logger.info(f'Parent groups: {len(output_df.groupby("parent"))}')
            logger.info(f'Execution time: {time.time() - start_time:.2f} seconds')
            logger.info(f'Output file: submission_task{task_level}.csv')
            logger.info('Status: ✓ SUCCESS')
            logger.info('='*80)
        else:
            logger.error(f'Task {task_level} validation failed! Continuing to next task...')
        
        logger.info('')
    
    logger.info('')
    logger.info('='*80)
    logger.info('ALL TASKS COMPLETED')
    logger.info('='*80)
    logger.info('Check the result/ directory for output CSV files.')


if __name__ == '__main__':
    main()
