"""
Plotting script for Vehicle Steering Parameter Identification experiments.
Generates visualizations comparing different identification methods.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def load_results(run_dir):
    """Load results from a run directory"""
    results_path = os.path.join(run_dir, 'results.json')
    final_info_path = os.path.join(run_dir, 'final_info.json')

    results = None
    final_info = None

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)

    if os.path.exists(final_info_path):
        with open(final_info_path, 'r') as f:
            final_info = json.load(f)

    return results, final_info


def plot_parameter_comparison(all_results, save_path='parameter_comparison.png'):
    """
    Plot comparison of identified parameters vs true values across methods
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    run_labels = []
    true_cf = None
    true_cr = None

    # First pass: collect run labels and determine which methods exist
    for run_name, results in sorted(all_results.items()):
        if results is None:
            continue
        run_labels.append(run_name)

    if not run_labels:
        print("No valid results to plot")
        return

    # Collect data for each run, using NaN for missing methods
    cf_data = {'Least Squares': [], 'Neural Network': [], 'Attention NN': []}
    cr_data = {'Least Squares': [], 'Neural Network': [], 'Attention NN': []}

    for run_name in run_labels:
        results = all_results[run_name]

        if 'least_squares' in results:
            cf_data['Least Squares'].append(results['least_squares']['pred_Cf'])
            cr_data['Least Squares'].append(results['least_squares']['pred_Cr'])
            true_cf = results['least_squares']['true_Cf']
            true_cr = results['least_squares']['true_Cr']
        else:
            cf_data['Least Squares'].append(np.nan)
            cr_data['Least Squares'].append(np.nan)

        if 'neural_network' in results:
            cf_data['Neural Network'].append(results['neural_network']['pred_Cf'])
            cr_data['Neural Network'].append(results['neural_network']['pred_Cr'])
        else:
            cf_data['Neural Network'].append(np.nan)
            cr_data['Neural Network'].append(np.nan)

        if 'attention_nn' in results:
            cf_data['Attention NN'].append(results['attention_nn']['pred_Cf'])
            cr_data['Attention NN'].append(results['attention_nn']['pred_Cr'])
        else:
            cf_data['Attention NN'].append(np.nan)
            cr_data['Attention NN'].append(np.nan)

    # Filter out methods that have no data at all
    methods = [m for m in cf_data.keys() if not all(np.isnan(v) for v in cf_data[m])]
    colors = {'Least Squares': '#2ecc71', 'Neural Network': '#3498db',
              'Attention NN': '#9b59b6'}

    x = np.arange(len(run_labels))
    width = 0.2

    # Plot Cf comparison
    ax = axes[0]
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, cf_data[method], width, label=method,
               color=colors.get(method, '#999999'), alpha=0.8)

    if true_cf:
        ax.axhline(y=true_cf, color='red', linestyle='--', linewidth=2, label=f'True Cf = {true_cf}')

    ax.set_xlabel('Run')
    ax.set_ylabel('Front Cornering Stiffness Cf [N/rad]')
    ax.set_title('Front Cornering Stiffness Identification')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot Cr comparison
    ax = axes[1]
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, cr_data[method], width, label=method,
               color=colors.get(method, '#999999'), alpha=0.8)

    if true_cr:
        ax.axhline(y=true_cr, color='red', linestyle='--', linewidth=2, label=f'True Cr = {true_cr}')

    ax.set_xlabel('Run')
    ax.set_ylabel('Rear Cornering Stiffness Cr [N/rad]')
    ax.set_title('Rear Cornering Stiffness Identification')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_comparison(all_results, save_path='error_comparison.png'):
    """
    Plot identification error comparison across methods and runs
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    all_errors = {}

    for run_name, results in all_results.items():
        if results is None:
            continue

        if 'least_squares' in results:
            method = 'Least Squares'
            if method not in all_errors:
                all_errors[method] = []
                methods.append(method)
            all_errors[method].append(results['least_squares']['mean_error_percent'])

        if 'neural_network' in results:
            method = 'Neural Network'
            if method not in all_errors:
                all_errors[method] = []
                methods.append(method)
            all_errors[method].append(results['neural_network']['mean_error_percent'])

        if 'attention_nn' in results:
            method = 'Attention NN'
            if method not in all_errors:
                all_errors[method] = []
                methods.append(method)
            all_errors[method].append(results['attention_nn']['mean_error_percent'])

    if not methods:
        print("No valid results to plot")
        return

    # Box plot of errors
    data = [all_errors[m] for m in methods]
    colors = ['#2ecc71', '#3498db', '#9b59b6'][:len(methods)]

    bp = ax.boxplot(data, labels=methods, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Mean Parameter Error (%)')
    ax.set_title('Parameter Identification Error Comparison')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')

    # Add individual points
    for i, (method, errors) in enumerate(all_errors.items()):
        x = np.random.normal(i + 1, 0.04, size=len(errors))
        ax.scatter(x, errors, alpha=0.6, color='black', s=30, zorder=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(all_results, save_path='training_curves.png'):
    """
    Plot training loss curves for neural network methods (NN and Attention NN)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    has_nn_data = False
    has_attn_data = False

    for run_name, results in sorted(all_results.items()):
        if results is None:
            continue

        # Standard Neural Network training curves
        if 'neural_network' in results and 'train_losses' in results['neural_network']:
            has_nn_data = True
            train_losses = results['neural_network']['train_losses']
            val_losses = results['neural_network']['val_losses']
            epochs = range(1, len(train_losses) + 1)

            axes[0].plot(epochs, train_losses, label=f'{run_name} (train)', alpha=0.7)
            axes[0].plot(epochs, val_losses, '--', label=f'{run_name} (val)', alpha=0.7)

        # Attention NN training curves
        if 'attention_nn' in results and 'train_losses' in results['attention_nn']:
            has_attn_data = True
            train_losses = results['attention_nn']['train_losses']
            val_losses = results['attention_nn']['val_losses']
            epochs = range(1, len(train_losses) + 1)

            axes[1].plot(epochs, train_losses, label=f'{run_name} (train)', alpha=0.7)
            axes[1].plot(epochs, val_losses, '--', label=f'{run_name} (val)', alpha=0.7)

    # Configure Standard NN plot
    if has_nn_data:
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Standard Neural Network Training')
        axes[0].legend(fontsize=7, loc='upper right')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No Standard NN training data',
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Standard Neural Network Training')

    # Configure Attention NN plot
    if has_attn_data:
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Attention-Enhanced NN Training')
        axes[1].legend(fontsize=7, loc='upper right')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Attention NN training data',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Attention-Enhanced NN Training')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_weights(all_results, save_path='attention_weights.png'):
    """
    Plot final attention weights as bar chart for each run
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    feature_names = ['delta', 'velocity', 'beta', 'yaw_rate']

    has_data = False
    run_names = []
    all_weights = []

    for run_name, results in sorted(all_results.items()):
        if results is None:
            continue

        if 'attention_nn' in results and 'final_attention_weights' in results['attention_nn']:
            has_data = True
            run_names.append(run_name)
            all_weights.append(results['attention_nn']['final_attention_weights'])

    if has_data:
        x = np.arange(len(feature_names))
        width = 0.8 / len(run_names) if len(run_names) > 0 else 0.15

        for i, (run_name, weights) in enumerate(zip(run_names, all_weights)):
            offset = (i - len(run_names)/2 + 0.5) * width
            ax.bar(x + offset, weights, width, label=run_name, alpha=0.8)

        ax.set_xlabel('Input Feature')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Final Attention Weights by Feature')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'No attention weight data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Final Attention Weights by Feature')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_noise_sensitivity(all_results, save_path='noise_sensitivity.png'):
    """
    Plot how identification error varies with noise level
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect data with noise levels for each method
    ls_data = []
    nn_data = []
    attn_data = []

    for run_name, results in all_results.items():
        if results is None or 'summary' not in results:
            continue

        noise = results['summary'].get('noise_level', None)
        if noise is None:
            continue

        if 'least_squares' in results:
            ls_data.append((noise, results['least_squares']['mean_error_percent']))
        if 'neural_network' in results:
            nn_data.append((noise, results['neural_network']['mean_error_percent']))
        if 'attention_nn' in results:
            attn_data.append((noise, results['attention_nn']['mean_error_percent']))

    # Sort each method's data by noise level
    if ls_data:
        ls_data.sort(key=lambda x: x[0])
        ls_noise, ls_errors = zip(*ls_data)
        ax.plot(ls_noise, ls_errors, 'o-', label='Least Squares', color='#2ecc71', markersize=8)

    if nn_data:
        nn_data.sort(key=lambda x: x[0])
        nn_noise, nn_errors = zip(*nn_data)
        ax.plot(nn_noise, nn_errors, 's-', label='Neural Network', color='#3498db', markersize=8)

    if attn_data:
        attn_data.sort(key=lambda x: x[0])
        attn_noise, attn_errors = zip(*attn_data)
        ax.plot(attn_noise, attn_errors, '^-', label='Attention NN', color='#9b59b6', markersize=8)

    if ls_data or nn_data or attn_data:
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Mean Parameter Error (%)')
        ax.set_title('Parameter Identification Error vs Measurement Noise')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for noise sensitivity plot',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main plotting function"""
    # Find all run directories (only directories, not .py files)
    run_dirs = sorted([d for d in glob.glob('run_*') if os.path.isdir(d)])

    if not run_dirs:
        print("No run directories found!")
        return

    print(f"Found {len(run_dirs)} run directories")

    # Load all results
    all_results = {}
    for run_dir in run_dirs:
        results, final_info = load_results(run_dir)
        if results is not None:
            all_results[run_dir] = results
            print(f"  Loaded: {run_dir}")
        else:
            print(f"  Skipped (no results): {run_dir}")

    if not all_results:
        print("No valid results found!")
        return

    # Generate plots
    print("\nGenerating plots...")
    plot_parameter_comparison(all_results)
    plot_error_comparison(all_results)
    plot_training_curves(all_results)
    plot_noise_sensitivity(all_results)
    plot_attention_weights(all_results)

    print("\nDone!")


if __name__ == '__main__':
    main()
