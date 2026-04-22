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
    'task1_current': 'current-prediction/l1-weights.csv',

    'task2_repos': 'data/level 2/repos_to_predict.csv',
    'task2_current': 'data/level 2/originality-predictions.csv',

    'task3_input': 'data/level 3/pairs_to_predict.csv',
    'task3_current': 'data/level 3/l2-predictions-example.csv',

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


class OriginalityPredictor:
    """
    Predicts originality score (0-1) for each repo.
    Score = how much of the repo's value is its own work vs its dependencies.
    0.2 = mostly fork/wrapper, 0.5 = balanced, 0.8 = mostly original work.
    """
    # Orgs known for highly original core infrastructure
    HIGH_ORIGINALITY_ORGS = {
        'ethereum', 'ethers-io', 'foundry-rs', 'paradigmxyz', 'sigp',
        'nomicfoundation', 'vyperlang', 'erigontech', 'alloy-rs', 'bluealloy',
        'argotorg', 'ipsilon', 'supranational', 'herumi', 'arkworks-rs',
        'espressosystems', 'plonky3', 'powdr-labs', 'lambdaclass',
    }
    # Orgs that build on top of others (middleware, tooling, wrappers)
    MID_ORIGINALITY_ORGS = {
        'openzeppelin', 'consensys', 'hyperledger', 'safe-global', 'wevm',
        'chainsafe', 'nethermindeth', 'flashbots', 'offchainlabs', 'status-im',
        'libp2p', 'scaffold-eth', 'eth-infinitism', 'protofire', 'blockscout',
        'defillama', 'l2beat', 'taikoxyz', 'risc0', 'succinctlabs',
    }

    def predict_originality(self, repos_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute originality score for each repo using:
        1. Org reputation tier (core infra = more original)
        2. Dependency count (more deps = less original)
        3. Whether repo is depended upon by others (being a dependency = more original)
        """
        results = []

        # Build dependency graph stats from pairs_to_predict
        dep_count = pairs_df.groupby('repo').size().to_dict()
        depended_on = pairs_df.groupby('dependency').size().to_dict()
        max_deps = max(dep_count.values()) if dep_count else 1
        max_dependents = max(depended_on.values()) if depended_on else 1

        for _, row in repos_df.iterrows():
            url = row['repo']
            url_lower = url.lower()

            # Feature 1: org tier base score (continuous)
            org = self._get_org(url_lower)
            if org in self.HIGH_ORIGINALITY_ORGS:
                tier_score = 0.72
            elif org in self.MID_ORIGINALITY_ORGS:
                tier_score = 0.52
            else:
                tier_score = 0.44

            # Feature 2: dependency count penalty (continuous, normalized)
            repo_short = self._to_short(url_lower)
            n_deps = dep_count.get(repo_short, dep_count.get(url_lower, 0))
            # Penalty scales from 0 to 0.25 based on normalized dep count
            dep_penalty = 0.25 * (n_deps / max_deps)

            # Feature 3: being depended upon bonus (continuous, normalized)
            n_dependents = depended_on.get(repo_short, depended_on.get(url_lower, 0))
            # Bonus scales from 0 to 0.15 based on normalized dependent count
            dep_bonus = 0.15 * (n_dependents / max_dependents)

            score = tier_score - dep_penalty + dep_bonus
            score = max(0.10, min(0.95, score))

            results.append({'repo': url, 'originality': round(score, 6)})

        return pd.DataFrame(results)

    def _get_org(self, url: str) -> str:
        match = re.search(r'github\.com/([^/]+)/', url)
        if match:
            return match.group(1).lower()
        parts = url.split('/')
        return parts[0].lower() if parts else 'unknown'

    def _to_short(self, url: str) -> str:
        match = re.search(r'github\.com/(.+)', url)
        return match.group(1).rstrip('/') if match else url


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
            # Task 2: just load the list of repos to predict originality for
            repos_df = pd.read_csv(self.config['task2_repos'])
            self.logger.info(f'Loaded {len(repos_df)} repos for originality prediction')
            return repos_df
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
        output_path = Path(self.config['output_dir']) / filename

        if level == 2:
            # Task 2: repo,originality format
            df_export['originality'] = df_export['originality'].apply(lambda x: f'{float(x):.6f}')
            df_export.to_csv(output_path, index=False, columns=['repo', 'originality'])
        elif level == 3:
            # Task 3: dependency,repo,weight format
            df_export['weight'] = df_export['weight'].apply(lambda x: f'{x:.10f}')
            df_export.to_csv(output_path, index=False, columns=['dependency', 'repo', 'weight'])
        else:
            # Task 1: repo,parent,weight format
            df_export['weight'] = df_export['weight'].apply(lambda x: f'{x:.10f}')
            df_export.to_csv(output_path, index=False, columns=['repo', 'parent', 'weight'])

        self.logger.info(f'✓ Exported {len(df_export)} rows to {filename}')
    
    def validate_output(self, df, input_df=None, level=None):
        self.logger.info('Starting output validation')
        validation_passed = True

        if df is None or len(df) == 0:
            self.logger.error('Validation failed: Empty DataFrame')
            return False

        if level == 2:
            # Task 2: repo,originality — each value in (0, 1)
            if not all(col in df.columns for col in ['repo', 'originality']):
                self.logger.error('Validation failed: Missing columns (repo, originality)')
                return False
            df['originality'] = pd.to_numeric(df['originality'], errors='coerce')
            invalid = df[(df['originality'] <= 0) | (df['originality'] >= 1)]
            if len(invalid) > 0:
                self.logger.error(f'Validation failed: {len(invalid)} originality values outside (0, 1)')
                validation_passed = False
            dups = df.duplicated(subset=['repo'], keep=False)
            if dups.any():
                self.logger.error(f'Validation failed: {dups.sum()} duplicate repos')
                validation_passed = False
        elif level == 3:
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

        # Task 1: use current-prediction/l1-weights.csv
        if level == 1:
            curr_path = self.config.get('task1_current', 'current-prediction/l1-weights.csv')
            try:
                curr_df = pd.read_csv(curr_path)
                def to_full_url(repo):
                    if not str(repo).startswith('http'):
                        return f'https://github.com/{repo}'
                    return repo
                curr_df['repo'] = curr_df['repo'].apply(to_full_url)
                if 'parent' not in curr_df.columns:
                    curr_df['parent'] = 'ethereum'
                # Normalize weights to sum exactly to 1.0
                total = curr_df['weight'].sum()
                curr_df['weight'] = curr_df['weight'] / total
                self.logger.info(f'Task 1: using current predictions ({len(curr_df)} rows, sum={curr_df["weight"].sum():.10f})')
                return curr_df
            except Exception as e:
                self.logger.warning(f'Could not load l1-weights.csv: {e}, falling back')

        # Task 2: use current-prediction/originality-weights.csv
        if level == 2:
            curr_path = self.config.get('task2_current', 'data/level 2/originality-predictions.csv')
            try:
                curr_df = pd.read_csv(curr_path)
                def to_full_url(repo):
                    if not str(repo).startswith('http'):
                        return f'https://github.com/{repo}'
                    return repo
                curr_df['repo'] = curr_df['repo'].apply(to_full_url)
                self.logger.info(f'Task 2: using current predictions ({len(curr_df)} rows)')
                return curr_df
            except Exception as e:
                self.logger.warning(f'Could not load originality-weights.csv: {e}, falling back')

        # Task 3: use current-prediction/l2-weights.csv
        if level == 3:
            curr_path = self.config.get('task3_current', 'data/level 3/l2-predictions-example.csv')
            try:
                curr_df = pd.read_csv(curr_path)
                def to_full_url(u):
                    if not str(u).startswith('http'):
                        return f'https://github.com/{u}'
                    return u
                curr_df['repo'] = curr_df['repo'].apply(to_full_url)
                curr_df['dependency'] = curr_df['dependency'].apply(to_full_url)
                # Normalize weights to sum exactly to 1.0 per repo
                curr_df['weight'] = curr_df.groupby('repo')['weight'].transform(lambda w: w / w.sum())
                self.logger.info(f'Task 3: using current predictions ({len(curr_df)} rows)')
                return curr_df
            except Exception as e:
                self.logger.warning(f'Could not load l2-weights.csv: {e}, falling back')

        # Build scores dict based on task level (Task 1 fallback only)
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
                logger.info(f'Repos with originality scores: {len(output_df)}')
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
