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

    methods = ['Least Squares', 'Neural Network']
    colors = ['#2ecc71', '#3498db']
    run_labels = []

    # Collect data for each run
    cf_data = {m: [] for m in methods}
    cr_data = {m: [] for m in methods}
    true_cf = None
    true_cr = None

    for run_name, results in all_results.items():
        if results is None:
            continue

        run_labels.append(run_name)

        if 'least_squares' in results:
            cf_data['Least Squares'].append(results['least_squares']['pred_Cf'])
            cr_data['Least Squares'].append(results['least_squares']['pred_Cr'])
            true_cf = results['least_squares']['true_Cf']
            true_cr = results['least_squares']['true_Cr']

        if 'neural_network' in results:
            cf_data['Neural Network'].append(results['neural_network']['pred_Cf'])
            cr_data['Neural Network'].append(results['neural_network']['pred_Cr'])

        if 'pinn' in results:
            if 'PINN' not in methods:
                methods.append('PINN')
                colors.append('#e74c3c')
                cf_data['PINN'] = []
                cr_data['PINN'] = []
            cf_data['PINN'].append(results['pinn']['pred_Cf'])
            cr_data['PINN'].append(results['pinn']['pred_Cr'])

    if not run_labels:
        print("No valid results to plot")
        return

    x = np.arange(len(run_labels))
    width = 0.25

    # Plot Cf comparison
    ax = axes[0]
    for i, method in enumerate(methods):
        if cf_data[method]:
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, cf_data[method], width, label=method, color=colors[i], alpha=0.8)

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
        if cr_data[method]:
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, cr_data[method], width, label=method, color=colors[i], alpha=0.8)

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

        if 'pinn' in results:
            method = 'PINN'
            if method not in all_errors:
                all_errors[method] = []
                methods.append(method)
            all_errors[method].append(results['pinn']['mean_error_percent'])

    if not methods:
        print("No valid results to plot")
        return

    # Box plot of errors
    data = [all_errors[m] for m in methods]
    colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(methods)]

    bp = ax.boxplot(data, labels=methods, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Mean Parameter Error (%)')
    ax.set_title('Parameter Identification Error Comparison')
    ax.grid(axis='y', alpha=0.3)

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
    Plot training loss curves for neural network methods
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    has_nn_data = False
    has_pinn_data = False

    for run_name, results in all_results.items():
        if results is None:
            continue

        # Neural Network training curves
        if 'neural_network' in results and 'train_losses' in results['neural_network']:
            has_nn_data = True
            train_losses = results['neural_network']['train_losses']
            val_losses = results['neural_network']['val_losses']
            epochs = range(1, len(train_losses) + 1)

            axes[0].plot(epochs, train_losses, label=f'{run_name} (train)', alpha=0.7)
            axes[0].plot(epochs, val_losses, '--', label=f'{run_name} (val)', alpha=0.7)

        # PINN training curves
        if 'pinn' in results and 'losses' in results['pinn']:
            has_pinn_data = True
            losses = results['pinn']['losses']
            param_errors = results['pinn']['param_errors']
            epochs = range(1, len(losses) + 1)

            axes[1].plot(epochs, param_errors, label=f'{run_name}', alpha=0.7)

    # Configure NN plot
    if has_nn_data:
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Neural Network Training Curves')
        axes[0].legend(fontsize=8)
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No NN training data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Neural Network Training Curves')

    # Configure PINN plot
    if has_pinn_data:
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Parameter Error (%)')
        axes[1].set_title('PINN Parameter Error During Training')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No PINN training data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('PINN Parameter Error During Training')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_noise_sensitivity(all_results, save_path='noise_sensitivity.png'):
    """
    Plot how identification error varies with noise level
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    noise_levels = []
    ls_errors = []
    nn_errors = []
    pinn_errors = []

    for run_name, results in all_results.items():
        if results is None or 'summary' not in results:
            continue

        noise = results['summary'].get('noise_level', None)
        if noise is None:
            continue

        noise_levels.append(noise)

        if 'least_squares' in results:
            ls_errors.append(results['least_squares']['mean_error_percent'])
        if 'neural_network' in results:
            nn_errors.append(results['neural_network']['mean_error_percent'])
        if 'pinn' in results:
            pinn_errors.append(results['pinn']['mean_error_percent'])

    if noise_levels:
        # Sort by noise level
        sorted_idx = np.argsort(noise_levels)
        noise_levels = [noise_levels[i] for i in sorted_idx]

        if ls_errors:
            ls_errors = [ls_errors[i] for i in sorted_idx]
            ax.plot(noise_levels, ls_errors, 'o-', label='Least Squares', color='#2ecc71', markersize=8)

        if nn_errors:
            nn_errors = [nn_errors[i] for i in sorted_idx]
            ax.plot(noise_levels, nn_errors, 's-', label='Neural Network', color='#3498db', markersize=8)

        if pinn_errors:
            pinn_errors = [pinn_errors[i] for i in sorted_idx]
            ax.plot(noise_levels, pinn_errors, '^-', label='PINN', color='#e74c3c', markersize=8)

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
    # Find all run directories
    run_dirs = sorted(glob.glob('run_*'))

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

    print("\nDone!")


if __name__ == '__main__':
    main()
