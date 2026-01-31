"""
Run multiple independent experiments for statistical reliability.

This script runs the parameter identification experiment multiple times for each noise level
with different random seeds to ensure statistical significance of the results.

Usage:
    python run_multiple_experiments.py --num_repeats 5 --base_seed 42
"""

import subprocess
import json
import os
import argparse
import numpy as np
from pathlib import Path


def run_single_experiment(noise_level, seed, run_index, base_dir='runs_multi'):
    """
    Run a single experiment with specified noise level and seed

    Args:
        noise_level: Measurement noise level (e.g., 0.01, 0.02, 0.05, 0.1)
        seed: Random seed for reproducibility
        run_index: Index of this run for naming
        base_dir: Base directory for output

    Returns:
        Path to the output directory
    """
    out_dir = os.path.join(base_dir, f'noise{noise_level}_run{run_index}_seed{seed}')

    cmd = [
        'python', 'experiment.py',
        '--out_dir', out_dir,
        '--seed', str(seed),
        '--noise_level', str(noise_level),
        '--num_samples', '5000',
        '--epochs', '100',
        '--lr', '0.001',
        '--batch_size', '64',
        '--hidden_sizes', '64,64',
        '--activation', 'relu'
    ]

    print(f"\n{'='*60}")
    print(f"Running experiment: noise={noise_level}, seed={seed}, run={run_index}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Experiment failed!")
        print(f"STDERR: {result.stderr}")
        return None

    print(f"✓ Completed successfully")
    return out_dir


def aggregate_results(noise_levels, num_repeats, base_dir='runs_multi'):
    """
    Aggregate results from multiple runs

    Args:
        noise_levels: List of noise levels tested
        num_repeats: Number of repetitions per noise level
        base_dir: Base directory containing all runs

    Returns:
        Dictionary with aggregated statistics
    """
    aggregated = {}

    for noise in noise_levels:
        # Collect results for this noise level
        ls_errors = []
        nn_errors = []
        attn_errors = []

        ls_cf_errors = []
        ls_cr_errors = []
        nn_cf_errors = []
        nn_cr_errors = []
        attn_cf_errors = []
        attn_cr_errors = []

        nn_pred_cf = []
        nn_pred_cr = []
        attn_pred_cf = []
        attn_pred_cr = []
        ls_pred_cf = []
        ls_pred_cr = []

        for run_idx in range(num_repeats):
            # Find the directory for this run
            pattern = os.path.join(base_dir, f'noise{noise}_run{run_idx}_*')
            import glob
            matching_dirs = glob.glob(pattern)

            if not matching_dirs:
                print(f"WARNING: No results found for noise={noise}, run={run_idx}")
                continue

            results_file = os.path.join(matching_dirs[0], 'results.json')

            if not os.path.exists(results_file):
                print(f"WARNING: Results file not found: {results_file}")
                continue

            with open(results_file, 'r') as f:
                results = json.load(f)

            # Extract errors
            if 'least_squares' in results:
                ls_errors.append(results['least_squares']['mean_error_percent'])
                ls_cf_errors.append(results['least_squares']['cf_error_percent'])
                ls_cr_errors.append(results['least_squares']['cr_error_percent'])
                ls_pred_cf.append(results['least_squares']['pred_Cf'])
                ls_pred_cr.append(results['least_squares']['pred_Cr'])

            if 'neural_network' in results:
                nn_errors.append(results['neural_network']['mean_error_percent'])
                nn_cf_errors.append(results['neural_network']['cf_error_percent'])
                nn_cr_errors.append(results['neural_network']['cr_error_percent'])
                nn_pred_cf.append(results['neural_network']['pred_Cf'])
                nn_pred_cr.append(results['neural_network']['pred_Cr'])

            if 'attention_nn' in results:
                attn_errors.append(results['attention_nn']['mean_error_percent'])
                attn_cf_errors.append(results['attention_nn']['cf_error_percent'])
                attn_cr_errors.append(results['attention_nn']['cr_error_percent'])
                attn_pred_cf.append(results['attention_nn']['pred_Cf'])
                attn_pred_cr.append(results['attention_nn']['pred_Cr'])

        # Compute statistics
        aggregated[noise] = {
            'least_squares': {
                'mean_error': {'mean': np.mean(ls_errors), 'std': np.std(ls_errors)},
                'cf_error': {'mean': np.mean(ls_cf_errors), 'std': np.std(ls_cf_errors)},
                'cr_error': {'mean': np.mean(ls_cr_errors), 'std': np.std(ls_cr_errors)},
                'pred_Cf': {'mean': np.mean(ls_pred_cf), 'std': np.std(ls_pred_cf)},
                'pred_Cr': {'mean': np.mean(ls_pred_cr), 'std': np.std(ls_pred_cr)},
                'n_samples': len(ls_errors)
            },
            'neural_network': {
                'mean_error': {'mean': np.mean(nn_errors), 'std': np.std(nn_errors)},
                'cf_error': {'mean': np.mean(nn_cf_errors), 'std': np.std(nn_cf_errors)},
                'cr_error': {'mean': np.mean(nn_cr_errors), 'std': np.std(nn_cr_errors)},
                'pred_Cf': {'mean': np.mean(nn_pred_cf), 'std': np.std(nn_pred_cf)},
                'pred_Cr': {'mean': np.mean(nn_pred_cr), 'std': np.std(nn_pred_cr)},
                'n_samples': len(nn_errors)
            },
            'attention_nn': {
                'mean_error': {'mean': np.mean(attn_errors), 'std': np.std(attn_errors)},
                'cf_error': {'mean': np.mean(attn_cf_errors), 'std': np.std(attn_cf_errors)},
                'cr_error': {'mean': np.mean(attn_cr_errors), 'std': np.std(attn_cr_errors)},
                'pred_Cf': {'mean': np.mean(attn_pred_cf), 'std': np.std(attn_pred_cf)},
                'pred_Cr': {'mean': np.mean(attn_pred_cr), 'std': np.std(attn_pred_cr)},
                'n_samples': len(attn_errors)
            }
        }

    return aggregated


def print_summary(aggregated):
    """Print a formatted summary of the aggregated results"""
    print("\n" + "="*80)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*80)

    for noise in sorted(aggregated.keys()):
        print(f"\nNoise Level: {noise}")
        print("-" * 80)

        data = aggregated[noise]

        print(f"{'Method':<20} {'Mean Error (%)':<25} {'Cf Error (%)':<25} {'Cr Error (%)':<25}")
        print("-" * 80)

        for method_name, method_label in [
            ('least_squares', 'Least Squares'),
            ('neural_network', 'Standard NN'),
            ('attention_nn', 'Attention NN')
        ]:
            if method_name in data and data[method_name]['n_samples'] > 0:
                mean_err = data[method_name]['mean_error']
                cf_err = data[method_name]['cf_error']
                cr_err = data[method_name]['cr_error']
                n = data[method_name]['n_samples']

                print(f"{method_label:<20} "
                      f"{mean_err['mean']:>8.5f} ± {mean_err['std']:<12.5f} "
                      f"{cf_err['mean']:>8.5f} ± {cf_err['std']:<12.5f} "
                      f"{cr_err['mean']:>8.5f} ± {cr_err['std']:<12.5f} "
                      f"(n={n})")

    print("\n" + "="*80)


def generate_latex_table(aggregated, output_file='aggregated_table.tex'):
    """Generate LaTeX table from aggregated results"""

    noise_levels = sorted(aggregated.keys())

    latex_lines = [
        "\\begin{table}[t]",
        "\\small\\sf\\centering",
        "\\caption{Parameter identification errors (\\%) across different noise levels. "
        "Results are averaged over 5 independent runs with different random seeds. "
        "Values shown as mean $\\pm$ standard deviation. Lower values indicate better performance.}",
        "\\label{tab:results}",
        "\\begin{tabular}{l" + "c" * len(noise_levels) + "}",
        "\\toprule",
    ]

    # Header row
    header = "\\textbf{Method}"
    for noise in noise_levels:
        header += f" & $\\eta = {noise}$"
    header += " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # Data rows
    for method_name, method_label in [
        ('least_squares', 'Least Squares'),
        ('neural_network', 'Standard NN'),
        ('attention_nn', 'Attention NN')
    ]:
        row = method_label
        for noise in noise_levels:
            if method_name in aggregated[noise]:
                mean_err = aggregated[noise][method_name]['mean_error']
                row += f" & ${mean_err['mean']:.3f} \\pm {mean_err['std']:.3f}$"
            else:
                row += " & ---"
        row += " \\\\"
        latex_lines.append(row)

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    latex_table = "\n".join(latex_lines)

    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\nLaTeX table saved to: {output_file}")
    print("\n" + "="*80)
    print("LaTeX Table Preview:")
    print("="*80)
    print(latex_table)
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple parameter identification experiments for statistical reliability'
    )
    parser.add_argument('--noise_levels', type=str, default='0.01,0.02,0.05,0.1',
                        help='Comma-separated noise levels to test (default: 0.01,0.02,0.05,0.1)')
    parser.add_argument('--num_repeats', type=int, default=5,
                        help='Number of repetitions per noise level (default: 5)')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--base_dir', type=str, default='runs_multi',
                        help='Base directory for outputs (default: runs_multi)')
    parser.add_argument('--skip_running', action='store_true',
                        help='Skip running experiments, only aggregate existing results')

    args = parser.parse_args()

    # Parse noise levels
    noise_levels = [float(x.strip()) for x in args.noise_levels.split(',')]

    # Generate seeds (use different seeds for each run)
    # Common good seeds: 42, 123, 456, 789, 1024, 2048, 3141, 5678, 9999, 12345
    seed_pool = [42, 123, 456, 789, 1024, 2048, 3141, 5678, 9999, 12345]
    seeds = seed_pool[:args.num_repeats]

    print(f"\n{'='*80}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Noise levels: {noise_levels}")
    print(f"Repetitions per noise level: {args.num_repeats}")
    print(f"Random seeds: {seeds}")
    print(f"Total experiments to run: {len(noise_levels) * args.num_repeats}")
    print(f"Output directory: {args.base_dir}")
    print(f"{'='*80}\n")

    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)

    # Run experiments
    if not args.skip_running:
        total_experiments = len(noise_levels) * args.num_repeats
        completed = 0

        for noise in noise_levels:
            for run_idx, seed in enumerate(seeds):
                completed += 1
                print(f"\n[Progress: {completed}/{total_experiments}]")

                out_dir = run_single_experiment(noise, seed, run_idx, args.base_dir)

                if out_dir is None:
                    print(f"ERROR: Experiment failed, continuing with next one...")
                    continue
    else:
        print("Skipping experiment execution (--skip_running flag set)")

    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)

    aggregated = aggregate_results(noise_levels, args.num_repeats, args.base_dir)

    # Save aggregated results
    aggregated_file = os.path.join(args.base_dir, 'aggregated_results.json')
    with open(aggregated_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            return obj

        json.dump(convert_numpy(aggregated), f, indent=2)

    print(f"\nAggregated results saved to: {aggregated_file}")

    # Print summary
    print_summary(aggregated)

    # Generate LaTeX table
    latex_file = os.path.join(args.base_dir, 'table_results.tex')
    generate_latex_table(aggregated, latex_file)

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"\nResults saved in: {args.base_dir}/")
    print(f"  - Individual runs: {args.base_dir}/noise*_run*_seed*/")
    print(f"  - Aggregated statistics: {aggregated_file}")
    print(f"  - LaTeX table: {latex_file}")
    print("\nNext steps:")
    print("  1. Review the aggregated results above")
    print("  2. Run the improved plotting script: python plot_multi.py")
    print("  3. Update your LaTeX paper with the new table and figures")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
