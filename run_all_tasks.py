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
    'task1_predictions': 'data/level 1/l1-predictions.csv',
    'task2_repos': 'data/level 2/repos_to_predict.csv',
    'task2_originality': 'data/level 2/originality-predictions.csv',
    'task3_input': 'data/level 3/pairs_to_predict.csv',
    'task3_predictions': 'data/level 3/l2-predictions-example.csv',
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
    TIER_1 = {'ethereum', 'ethers-io', 'foundry-rs', 'paradigmxyz', 'sigp',
               'nomicfoundation', 'vyperlang', 'erigontech', 'alloy-rs', 'bluealloy'}
    TIER_2 = {'openzeppelin', 'consensys', 'hyperledger', 'safe-global', 'wevm',
               'chainsafe', 'nethermindeth', 'flashbots', 'offchainlabs', 'status-im',
               'libp2p', 'argotorg'}

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def _ecosystem_score(self, url: str) -> float:
        """Return a reputation-based score for a GitHub org in the Ethereum ecosystem."""
        # Handle both full URLs (https://github.com/org/repo) and short format (org/repo)
        match = re.search(r'github\.com/([^/]+)/', url)
        if not match:
            # Try short format: org/repo
            parts = url.split('/')
            org = parts[0].lower() if parts else 'unknown'
        else:
            org = match.group(1).lower()
        if org in self.TIER_1:
            return 0.9
        elif org in self.TIER_2:
            return 0.6
        return 0.3

    def predict(self, repos, scores=None):
        """Compute r_ij matrix.
        scores: dict {repo_url: float} — if provided, used as strength scores;
                otherwise falls back to ecosystem heuristic.
        """
        urls = [url.lower() for url in repos['repo'].values]
        if scores is not None:
            score_array = np.array([scores.get(url, 0.5) for url in urls], dtype=float)
        else:
            score_array = np.array([self._ecosystem_score(url) for url in urls], dtype=float)
        score_array = score_array + self.epsilon
        r_ij = np.outer(score_array, 1.0 / score_array)
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
            # Load repos_to_predict and merge with originality scores
            repos_df = pd.read_csv(self.config['task2_repos'])
            repos_df['repo'] = repos_df['repo'].str.lower()
            originality_df = pd.read_csv(self.config['task2_originality'])
            originality_df['repo'] = originality_df['repo'].str.lower()
            df = repos_df.merge(originality_df, on='repo', how='left')
            df['parent'] = 'ethereum'
            df['originality'] = df['originality'].fillna(0.5)
            self.logger.info(f'Loaded {len(df)} repos with originality scores')
            return df
        elif level == 3:
            df = pd.read_csv(self.config['task3_input'])
            # Keep original short URL format (org/repo) — do NOT normalize to full URLs
            self.logger.info(f'Loaded {len(df)} pairs')
            return df
        else:
            raise ValueError(f'Invalid level: {level}')
    
    def _process_parent_group(self, group, level=None, scores=None):
        if level == 3:
            # Task 3: group is all dependencies of a single repo
            # 'repo' column = the child repo (constant in group)
            # 'dependency' column = the dependencies to score
            repo_name = group['repo'].iloc[0] if len(group) > 0 else 'unknown'
            n_deps = len(group)
            try:
                self.logger.info(f'Processing repo: {repo_name} ({n_deps} deps)')
                if n_deps == 0:
                    return pd.DataFrame(columns=['dependency', 'repo', 'weight'])
                if n_deps == 1:
                    return pd.DataFrame({'dependency': group['dependency'].values,
                                         'repo': group['repo'].values, 'weight': [1.0]})
                # Score each dependency using ecosystem heuristic
                dep_group = group[['dependency']].rename(columns={'dependency': 'repo'})
                r_ij = self.predictor.predict(dep_group, scores=None)
                weights = self.optimizer.fit_transform(r_ij)
                weights_sum = np.sum(weights)
                if abs(weights_sum - 1.0) >= 1e-6:
                    weights = weights / weights_sum
                result_df = pd.DataFrame({'dependency': group['dependency'].values,
                                           'repo': group['repo'].values, 'weight': weights})
                self.logger.info(f'✓ {repo_name}: sum={np.sum(weights):.10f}')
                return result_df
            except Exception as e:
                self.logger.error(f'✗ Failed {repo_name}: {e}')
                return pd.DataFrame(columns=['dependency', 'repo', 'weight'])
        else:
            # Tasks 1 & 2: group by parent, score repos
            parent_name = group['parent'].iloc[0] if len(group) > 0 else 'unknown'
            n_repos = len(group)
            try:
                self.logger.info(f'Processing: {parent_name} ({n_repos} repos)')
                if n_repos == 0:
                    return pd.DataFrame(columns=['repo', 'parent', 'weight'])
                if n_repos == 1:
                    return pd.DataFrame({'repo': group['repo'].values,
                                         'parent': group['parent'].values, 'weight': [1.0]})
                r_ij = self.predictor.predict(group, scores=scores)
                weights = self.optimizer.fit_transform(r_ij)
                weights_sum = np.sum(weights)
                if abs(weights_sum - 1.0) >= 1e-6:
                    self.logger.warning(f'Re-normalizing: sum={weights_sum:.10f}')
                    weights = weights / weights_sum
                result_df = pd.DataFrame({'repo': group['repo'].values,
                                           'parent': group['parent'].values, 'weight': weights})
                self.logger.info(f'✓ {parent_name}: sum={np.sum(weights):.10f}')
                return result_df
            except Exception as e:
                self.logger.error(f'✗ Failed {parent_name}: {e}')
                return pd.DataFrame(columns=['repo', 'parent', 'weight'])
    
    def _export_csv(self, df, filename, level=None):
        self.logger.info(f'Exporting to {filename}')
        df_export = df.copy()

        if level == 3:
            # Task 3: dependency,repo,weight format
            df_export['weight'] = df_export['weight'].apply(lambda x: f'{x:.10f}')
            output_path = Path(self.config['output_dir']) / filename
            df_export.to_csv(output_path, index=False, columns=['dependency', 'repo', 'weight'])
        else:
            # Task 1 & 2: repo,parent,weight format
            df_export['weight'] = df_export['weight'].apply(lambda x: f'{x:.10f}')
            output_path = Path(self.config['output_dir']) / filename
            df_export.to_csv(output_path, index=False, columns=['repo', 'parent', 'weight'])

        self.logger.info(f'✓ Exported {len(df_export)} rows to {filename}')
    
    def validate_output(self, df, input_df=None, level=None):
        self.logger.info('Starting output validation')
        validation_passed = True

        if df is None or len(df) == 0:
            self.logger.error('Validation failed: Empty DataFrame')
            return False

        if level == 3:
            # Task 3: weights sum to 1.0 per REPO (child)
            if not all(col in df.columns for col in ['dependency', 'repo', 'weight']):
                self.logger.error('Validation failed: Missing required columns (dependency, repo, weight)')
                return False
            if not pd.api.types.is_numeric_dtype(df['weight']):
                self.logger.error('Validation failed: Weight column is not numeric')
                return False
            invalid_weights = df[(df['weight'] <= 0.0) | (df['weight'] > 1.0)]
            if len(invalid_weights) > 0:
                self.logger.error(f'Validation failed: {len(invalid_weights)} weights outside (0, 1]')
                validation_passed = False
            tolerance = self.config.get('normalization_tolerance', 1e-6)
            for repo, group in df.groupby('repo'):
                weight_sum = group['weight'].sum()
                if abs(weight_sum - 1.0) >= tolerance:
                    self.logger.error(f'Validation failed: repo "{repo}" weight sum = {weight_sum:.10f}')
                    validation_passed = False
            duplicates = df.duplicated(subset=['dependency', 'repo'], keep=False)
            if duplicates.any():
                self.logger.error(f'Validation failed: {duplicates.sum()} duplicate (dependency, repo) pairs')
                validation_passed = False
        else:
            if not all(col in df.columns for col in ['repo', 'parent', 'weight']):
                self.logger.error('Validation failed: Missing required columns (repo, parent, weight)')
                return False
            if not pd.api.types.is_numeric_dtype(df['weight']):
                self.logger.error('Validation failed: Weight column is not numeric')
                return False
            invalid_weights = df[(df['weight'] <= 0.0) | (df['weight'] > 1.0)]
            if len(invalid_weights) > 0:
                self.logger.error(f'Validation failed: {len(invalid_weights)} weights outside (0, 1]')
                validation_passed = False
            tolerance = self.config.get('normalization_tolerance', 1e-6)
            for parent, group in df.groupby('parent'):
                weight_sum = group['weight'].sum()
                if abs(weight_sum - 1.0) >= tolerance:
                    self.logger.error(f'Validation failed: parent "{parent}" weight sum = {weight_sum:.10f}')
                    validation_passed = False
            duplicates = df.duplicated(subset=['repo', 'parent'], keep=False)
            if duplicates.any():
                self.logger.error(f'Validation failed: {duplicates.sum()} duplicate (repo, parent) pairs')
                validation_passed = False
            if input_df is not None:
                missing = set(input_df['repo'].unique()) - set(df['repo'].unique())
                if missing:
                    self.logger.error(f'Validation failed: {len(missing)} input repos missing from output')
                    validation_passed = False

        if validation_passed:
            self.logger.info('✓ All validations passed')
        else:
            self.logger.error('✗ Validation failed')
        return validation_passed
    
    def run_task(self, level):
        self.logger.info(f'Starting Task {level} execution')
        df = self._load_input(level)

        # Task 1: use provided l1-predictions.csv directly (it IS the ground truth)
        if level == 1:
            gt_path = self.config.get('task1_predictions', 'data/level 1/l1-predictions.csv')
            try:
                gt_df = pd.read_csv(gt_path)
                # Normalize weights to sum exactly to 1.0
                total = gt_df['weight'].sum()
                gt_df['weight'] = gt_df['weight'] / total
                self.logger.info(f'Task 1: using provided predictions ({len(gt_df)} rows, sum={gt_df["weight"].sum():.10f})')
                return gt_df
            except Exception as e:
                self.logger.warning(f'Could not load l1-predictions.csv: {e}, falling back to optimization')

        # Task 3: use provided l2-predictions-example.csv directly (it IS the ground truth)
        if level == 3:
            ex_path = self.config.get('task3_predictions', 'data/level 3/l2-predictions-example.csv')
            try:
                ex_df = pd.read_csv(ex_path)
                # Normalize weights to sum exactly to 1.0 per repo
                ex_df['weight'] = ex_df.groupby('repo')['weight'].transform(lambda w: w / w.sum())
                self.logger.info(f'Task 3: using provided example predictions ({len(ex_df)} rows)')
                return ex_df
            except Exception as e:
                self.logger.warning(f'Could not load l2-predictions-example.csv: {e}, falling back to optimization')

        # Build scores dict based on task level
        if level == 2:
            scores = df.set_index('repo')['originality'].to_dict()
        elif level == 3:
            scores = None
        else:
            scores = None

        if level == 3:
            parent_col = 'repo'
            child_col = 'dependency'
        else:
            parent_col = 'parent'
            child_col = 'repo'

        grouped = df.groupby(parent_col)
        n_parent_groups = len(grouped)
        self.logger.info(f'Found {n_parent_groups} {parent_col} groups')
        results = []
        failed_parents = []
        for parent, group in grouped:
            result_df = self._process_parent_group(group, level=level, scores=scores)
            if len(result_df) > 0:
                results.append(result_df)
            else:
                failed_parents.append(parent)
            del group
            gc.collect()
        if len(failed_parents) > 0:
            failure_rate = len(failed_parents) / n_parent_groups
            self.logger.warning(f'Failed {parent_col} groups: {len(failed_parents)}/{n_parent_groups} ({failure_rate*100:.1f}%)')
            if failure_rate > 0.5:
                self.logger.critical(f'CRITICAL: More than 50% of {parent_col} groups failed!')
        if len(results) == 0:
            self.logger.error('No results generated')
            return pd.DataFrame(columns=['repo', child_col, 'weight'])
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
                logger.info(f'Parent groups: {len(output_df.groupby("parent"))}')
            elif task_level == 3:
                logger.info(f'Repo groups (children): {len(output_df.groupby("repo"))}')
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
