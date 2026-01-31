"""
Enhanced plotting script for multiple experimental runs.

This script generates publication-quality plots with error bars (mean ± std)
from multiple independent experimental runs.

Usage:
    python plot_multi.py --runs_dir runs_multi
"""

import json
import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def load_all_results(runs_dir):
    """
    Load results from all run directories

    Returns:
        Dictionary mapping noise level to list of results for that noise level
    """
    results_by_noise = defaultdict(list)

    # Find all run directories
    run_dirs = glob.glob(os.path.join(runs_dir, 'noise*_run*_seed*'))

    print(f"Found {len(run_dirs)} run directories in {runs_dir}")

    for run_dir in sorted(run_dirs):
        results_file = os.path.join(run_dir, 'results.json')

        if not os.path.exists(results_file):
            print(f"  Skipping {run_dir} (no results.json)")
            continue

        with open(results_file, 'r') as f:
            results = json.load(f)

        if 'summary' in results and 'noise_level' in results['summary']:
            noise = results['summary']['noise_level']
            results_by_noise[noise].append(results)
            print(f"  Loaded: {os.path.basename(run_dir)} (noise={noise})")
        else:
            print(f"  Skipping {run_dir} (no noise level info)")

    return results_by_noise


def plot_noise_sensitivity_with_errorbar(results_by_noise, save_path='noise_sensitivity.png'):
    """
    Plot identification error vs noise level with error bars (mean ± std)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    noise_levels = sorted(results_by_noise.keys())

    # Collect statistics for each method
    methods_data = {
        'Least Squares': {'noise': [], 'mean': [], 'std': [], 'color': '#2ecc71', 'marker': 'o'},
        'Neural Network': {'noise': [], 'mean': [], 'std': [], 'color': '#3498db', 'marker': 's'},
        'Attention NN': {'noise': [], 'mean': [], 'std': [], 'color': '#9b59b6', 'marker': '^'}
    }

    for noise in noise_levels:
        runs = results_by_noise[noise]

        # Collect errors from all runs at this noise level
        ls_errors = [r['least_squares']['mean_error_percent'] for r in runs if 'least_squares' in r]
        nn_errors = [r['neural_network']['mean_error_percent'] for r in runs if 'neural_network' in r]
        attn_errors = [r['attention_nn']['mean_error_percent'] for r in runs if 'attention_nn' in r]

        if ls_errors:
            methods_data['Least Squares']['noise'].append(noise)
            methods_data['Least Squares']['mean'].append(np.mean(ls_errors))
            methods_data['Least Squares']['std'].append(np.std(ls_errors))

        if nn_errors:
            methods_data['Neural Network']['noise'].append(noise)
            methods_data['Neural Network']['mean'].append(np.mean(nn_errors))
            methods_data['Neural Network']['std'].append(np.std(nn_errors))

        if attn_errors:
            methods_data['Attention NN']['noise'].append(noise)
            methods_data['Attention NN']['mean'].append(np.mean(attn_errors))
            methods_data['Attention NN']['std'].append(np.std(attn_errors))

    # Plot each method with error bars
    for method_name, data in methods_data.items():
        if data['noise']:
            ax.errorbar(
                data['noise'], data['mean'], yerr=data['std'],
                label=method_name,
                color=data['color'],
                marker=data['marker'],
                markersize=8,
                linewidth=2,
                capsize=5,
                capthick=2,
                alpha=0.8
            )

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Mean Parameter Error (%)', fontsize=12)
    ax.set_title('Parameter Identification Error vs Measurement Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_parameter_comparison_with_errorbar(results_by_noise, save_path='parameter_comparison.png'):
    """
    Plot predicted parameters with error bars across noise levels
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    noise_levels = sorted(results_by_noise.keys())

    # True values (from first result)
    first_result = list(results_by_noise.values())[0][0]
    true_cf = first_result['least_squares']['true_Cf']
    true_cr = first_result['least_squares']['true_Cr']

    methods_data = {
        'Least Squares': {'color': '#2ecc71'},
        'Neural Network': {'color': '#3498db'},
        'Attention NN': {'color': '#9b59b6'}
    }

    for method_key, method_name in [
        ('least_squares', 'Least Squares'),
        ('neural_network', 'Neural Network'),
        ('attention_nn', 'Attention NN')
    ]:
        cf_means, cf_stds = [], []
        cr_means, cr_stds = [], []

        for noise in noise_levels:
            runs = results_by_noise[noise]

            cf_values = [r[method_key]['pred_Cf'] for r in runs if method_key in r]
            cr_values = [r[method_key]['pred_Cr'] for r in runs if method_key in r]

            cf_means.append(np.mean(cf_values))
            cf_stds.append(np.std(cf_values))
            cr_means.append(np.mean(cr_values))
            cr_stds.append(np.std(cr_values))

        x = np.arange(len(noise_levels))
        width = 0.25
        offset = (list(methods_data.keys()).index(method_name) - 1) * width

        # Plot Cf
        axes[0].bar(
            x + offset, cf_means, width,
            yerr=cf_stds,
            label=method_name,
            color=methods_data[method_name]['color'],
            alpha=0.7,
            capsize=4
        )

        # Plot Cr
        axes[1].bar(
            x + offset, cr_means, width,
            yerr=cr_stds,
            label=method_name,
            color=methods_data[method_name]['color'],
            alpha=0.7,
            capsize=4
        )

    # Configure Cf subplot
    axes[0].axhline(y=true_cf, color='red', linestyle='--', linewidth=2, label=f'True $C_f$ = {true_cf}')
    axes[0].set_xlabel('Noise Level', fontsize=12)
    axes[0].set_ylabel('Front Cornering Stiffness $C_f$ [N/rad]', fontsize=12)
    axes[0].set_title('Front Cornering Stiffness Identification', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{n}' for n in noise_levels])
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    # Configure Cr subplot
    axes[1].axhline(y=true_cr, color='red', linestyle='--', linewidth=2, label=f'True $C_r$ = {true_cr}')
    axes[1].set_xlabel('Noise Level', fontsize=12)
    axes[1].set_ylabel('Rear Cornering Stiffness $C_r$ [N/rad]', fontsize=12)
    axes[1].set_title('Rear Cornering Stiffness Identification', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{n}' for n in noise_levels])
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_comparison_boxplot(results_by_noise, save_path='error_comparison.png'):
    """
    Box plot comparison of errors across all runs
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all errors across all noise levels
    all_errors = {
        'Least Squares': [],
        'Neural Network': [],
        'Attention NN': []
    }

    for noise, runs in results_by_noise.items():
        for result in runs:
            if 'least_squares' in result:
                all_errors['Least Squares'].append(result['least_squares']['mean_error_percent'])
            if 'neural_network' in result:
                all_errors['Neural Network'].append(result['neural_network']['mean_error_percent'])
            if 'attention_nn' in result:
                all_errors['Attention NN'].append(result['attention_nn']['mean_error_percent'])

    # Create box plot
    methods = list(all_errors.keys())
    data = [all_errors[m] for m in methods]
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    bp = ax.boxplot(data, labels=methods, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (method, errors) in enumerate(all_errors.items()):
        x = np.random.normal(i + 1, 0.04, size=len(errors))
        ax.scatter(x, errors, alpha=0.5, color='black', s=20, zorder=3)

    ax.set_ylabel('Mean Parameter Error (%)', fontsize=12)
    ax.set_title('Parameter Identification Error Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves_aggregated(results_by_noise, save_path='training_curves.png'):
    """
    Plot aggregated training curves (mean ± std shading)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect training curves for each noise level
    for noise in sorted(results_by_noise.keys()):
        runs = results_by_noise[noise]

        # Collect NN training curves
        nn_train_curves = []
        nn_val_curves = []

        # Collect Attention NN training curves
        attn_train_curves = []
        attn_val_curves = []

        for result in runs:
            if 'neural_network' in result and 'train_losses' in result['neural_network']:
                nn_train_curves.append(result['neural_network']['train_losses'])
                nn_val_curves.append(result['neural_network']['val_losses'])

            if 'attention_nn' in result and 'train_losses' in result['attention_nn']:
                attn_train_curves.append(result['attention_nn']['train_losses'])
                attn_val_curves.append(result['attention_nn']['val_losses'])

        # Plot NN curves
        if nn_train_curves:
            epochs = range(1, len(nn_train_curves[0]) + 1)
            train_mean = np.mean(nn_train_curves, axis=0)
            train_std = np.std(nn_train_curves, axis=0)
            val_mean = np.mean(nn_val_curves, axis=0)
            val_std = np.std(nn_val_curves, axis=0)

            axes[0].plot(epochs, train_mean, label=f'Train (noise={noise})', linewidth=2)
            axes[0].fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
            axes[0].plot(epochs, val_mean, '--', label=f'Val (noise={noise})', linewidth=2)
            axes[0].fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)

        # Plot Attention NN curves
        if attn_train_curves:
            epochs = range(1, len(attn_train_curves[0]) + 1)
            train_mean = np.mean(attn_train_curves, axis=0)
            train_std = np.std(attn_train_curves, axis=0)
            val_mean = np.mean(attn_val_curves, axis=0)
            val_std = np.std(attn_val_curves, axis=0)

            axes[1].plot(epochs, train_mean, label=f'Train (noise={noise})', linewidth=2)
            axes[1].fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
            axes[1].plot(epochs, val_mean, '--', label=f'Val (noise={noise})', linewidth=2)
            axes[1].fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)

    # Configure NN subplot
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Standard Neural Network Training', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # Configure Attention NN subplot
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Attention-Enhanced NN Training', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_weights_evolution(results_by_noise, save_path='attention_weights.png'):
    """
    Plot evolution of attention weights during training (averaged across runs)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    feature_names = ['delta', 'velocity', 'beta', 'yaw_rate']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    noise_levels = sorted(results_by_noise.keys())

    for noise_idx, noise in enumerate(noise_levels[:4]):  # Plot up to 4 noise levels
        ax = axes[noise_idx]
        runs = results_by_noise[noise]

        # Collect attention weight histories
        all_attention_histories = []

        for result in runs:
            if 'attention_nn' in result and 'attention_weights' in result['attention_nn']:
                attention_history = result['attention_nn']['attention_weights']
                all_attention_histories.append(np.array(attention_history))

        if all_attention_histories:
            # Average across runs
            mean_attention = np.mean(all_attention_histories, axis=0)
            std_attention = np.std(all_attention_histories, axis=0)

            epochs = range(1, len(mean_attention) + 1)

            # Plot each feature
            for feat_idx, (feat_name, color) in enumerate(zip(feature_names, colors)):
                feat_mean = mean_attention[:, feat_idx]
                feat_std = std_attention[:, feat_idx]

                ax.plot(epochs, feat_mean, label=feat_name, color=color, linewidth=2)
                ax.fill_between(epochs, feat_mean - feat_std, feat_mean + feat_std,
                                color=color, alpha=0.2)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Attention Weight', fontsize=11)
        ax.set_title(f'Noise Level = {noise}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)

    plt.suptitle('Evolution of Attention Weights During Training', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_attention_weights(results_by_noise, save_path='final_attention_weights.png'):
    """
    Plot final converged attention weights (mean ± std) for each noise level
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    feature_names = ['delta', 'velocity', 'beta', 'yaw_rate']
    noise_levels = sorted(results_by_noise.keys())

    x = np.arange(len(feature_names))
    width = 0.8 / len(noise_levels)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(noise_levels)))

    for noise_idx, noise in enumerate(noise_levels):
        runs = results_by_noise[noise]

        # Collect final attention weights
        final_weights = []

        for result in runs:
            if 'attention_nn' in result and 'final_attention_weights' in result['attention_nn']:
                final_weights.append(result['attention_nn']['final_attention_weights'])

        if final_weights:
            final_weights = np.array(final_weights)
            mean_weights = np.mean(final_weights, axis=0)
            std_weights = np.std(final_weights, axis=0)

            offset = (noise_idx - len(noise_levels)/2 + 0.5) * width

            ax.bar(
                x + offset, mean_weights, width,
                yerr=std_weights,
                label=f'$\\eta = {noise}$',
                color=colors[noise_idx],
                alpha=0.8,
                capsize=3
            )

    ax.set_xlabel('Input Feature', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Final Converged Attention Weights by Noise Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, fontsize=11)
    ax.legend(fontsize=10, title='Noise Level', title_fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance_comparison(results_by_noise, save_path='feature_importance_comparison.png'):
    """
    Plot comparison of feature importance between baseline and attention NN
    Matches the format described in the paper
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    feature_names = ['delta', 'velocity', 'beta', 'yaw_rate']

    # Baseline NN has implicit equal importance (0.25 each)
    baseline_weights = [0.25, 0.25, 0.25, 0.25]

    # Collect attention weights from all runs across all noise levels
    all_attention_weights = []
    for noise, runs in results_by_noise.items():
        for result in runs:
            if 'attention_nn' in result and 'final_attention_weights' in result['attention_nn']:
                all_attention_weights.append(result['attention_nn']['final_attention_weights'])

    if not all_attention_weights:
        print("No attention weights found")
        return

    # Compute mean and std of attention weights across all runs
    attention_weights = np.mean(all_attention_weights, axis=0)
    attention_std = np.std(all_attention_weights, axis=0)

    x = np.arange(len(feature_names))
    width = 0.35

    # Plot bars with error bars for attention weights
    ax.bar(x - width/2, baseline_weights, width, label='Baseline NN (Implicit)',
           color='#3498db', alpha=0.8)
    ax.bar(x + width/2, attention_weights, width, yerr=attention_std,
           label='Attention NN (Learned)', color='#9b59b6', alpha=0.8, capsize=5)

    ax.set_xlabel('Input Feature', fontsize=12)
    ax.set_ylabel('Feature Importance', fontsize=12)
    ax.set_title('Comparison of Feature Importance: Baseline vs Attention Mechanism', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add value labels on attention bars for significant values
    for i, (val, std) in enumerate(zip(attention_weights, attention_std)):
        if val > 0.01:  # Only show labels for significant values
            ax.text(x[i] + width/2, val + std + 0.02,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality plots from multiple experimental runs'
    )
    parser.add_argument('--runs_dir', type=str, default='runs_multi',
                        help='Directory containing all experimental runs (default: runs_multi)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save plots (default: current directory)')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"LOADING RESULTS FROM: {args.runs_dir}")
    print(f"{'='*80}\n")

    # Load all results
    results_by_noise = load_all_results(args.runs_dir)

    if not results_by_noise:
        print("ERROR: No results found!")
        return

    print(f"\n{'='*80}")
    print(f"GENERATING PLOTS")
    print(f"{'='*80}\n")

    # Generate all plots (matching paper figure references)
    plot_noise_sensitivity_with_errorbar(
        results_by_noise,
        os.path.join(args.output_dir, 'noise_sensitivity.png')
    )

    plot_final_attention_weights(
        results_by_noise,
        os.path.join(args.output_dir, 'final_attention_weights.png')
    )

    plot_feature_importance_comparison(
        results_by_noise,
        os.path.join(args.output_dir, 'feature_importance_comparison.png')
    )

    plot_parameter_comparison_with_errorbar(
        results_by_noise,
        os.path.join(args.output_dir, 'parameter_comparison.png')
    )

    plot_error_comparison_boxplot(
        results_by_noise,
        os.path.join(args.output_dir, 'error_comparison.png')
    )

    plot_training_curves_aggregated(
        results_by_noise,
        os.path.join(args.output_dir, 'training_curves.png')
    )

    # Optional: attention weights evolution (not directly referenced in paper but useful)
    plot_attention_weights_evolution(
        results_by_noise,
        os.path.join(args.output_dir, 'attention_weights.png')
    )

    print(f"\n{'='*80}")
    print(f"ALL PLOTS GENERATED!")
    print(f"{'='*80}")
    print(f"\nPlots saved in: {args.output_dir}/")
    print(f"  - noise_sensitivity.png (Fig: noise_sensitivity)")
    print(f"  - final_attention_weights.png (Fig: final_attention)")
    print(f"  - feature_importance_comparison.png (Fig: feature_importance)")
    print(f"  - parameter_comparison.png (Fig: parameter_comparison)")
    print(f"  - error_comparison.png (Fig: error_comparison)")
    print(f"  - training_curves.png (Fig: training_curves)")
    print(f"  - attention_weights.png (additional analysis)")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
