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

    methods = ['Least Squares', 'Neural Network', 'Attention Network', 'PINN']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
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

        # Determine pred values for each method, fill np.nan if missing
        for method in methods:
            if method == 'Least Squares':
                key = 'least_squares'
            elif method == 'Neural Network':
                key = 'neural_network'
            elif method == 'Attention Network':
                key = 'attention_network'
            elif method == 'PINN':
                key = 'pinn'
            else:
                key = None

            if key in results:
                cf_val = results[key]['pred_Cf']
                cr_val = results[key]['pred_Cr']
                # Store true values from any method (they are the same)
                if true_cf is None:
                    true_cf = results[key]['true_Cf']
                    true_cr = results[key]['true_Cr']
            else:
                cf_val = np.nan
                cr_val = np.nan
            cf_data[method].append(cf_val)
            cr_data[method].append(cr_val)

    if not run_labels:
        print("No valid results to plot")
        return

    x = np.arange(len(run_labels))
    width = 0.25

    # Plot Cf comparison
    ax = axes[0]
    for i, method in enumerate(methods):
        heights = cf_data[method]
        # Only plot where height is not nan
        valid_idx = [idx for idx, h in enumerate(heights) if not np.isnan(h)]
        if not valid_idx:
            continue
        valid_heights = [heights[idx] for idx in valid_idx]
        valid_x = [x[idx] for idx in valid_idx]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(np.array(valid_x) + offset, valid_heights, width, label=method, color=colors[i], alpha=0.8)

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
        heights = cr_data[method]
        valid_idx = [idx for idx, h in enumerate(heights) if not np.isnan(h)]
        if not valid_idx:
            continue
        valid_heights = [heights[idx] for idx in valid_idx]
        valid_x = [x[idx] for idx in valid_idx]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(np.array(valid_x) + offset, valid_heights, width, label=method, color=colors[i], alpha=0.8)

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

        if 'attention_network' in results:
            method = 'Attention Network'
            if method not in all_errors:
                all_errors[method] = []
                methods.append(method)
            all_errors[method].append(results['attention_network']['mean_error_percent'])

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

    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True)
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

        # Attention Network training curves
        if 'attention_network' in results and 'train_losses' in results['attention_network']:
            has_nn_data = True
            train_losses = results['attention_network']['train_losses']
            val_losses = results['attention_network']['val_losses']
            epochs = range(1, len(train_losses) + 1)

            axes[0].plot(epochs, train_losses, label=f'{run_name} attn (train)', alpha=0.7, linestyle='-.')
            axes[0].plot(epochs, val_losses, ':', label=f'{run_name} attn (val)', alpha=0.7)

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

    # Build list of records for each run
    records = []

    for run_name, results in all_results.items():
        if results is None or 'summary' not in results:
            continue

        noise = results['summary'].get('noise_level', None)
        if noise is None:
            continue

        record = {'noise': noise}

        if 'least_squares' in results:
            record['ls'] = results['least_squares']['mean_error_percent']
        if 'neural_network' in results:
            record['nn'] = results['neural_network']['mean_error_percent']
        if 'attention_network' in results:
            record['attn'] = results['attention_network']['mean_error_percent']
        if 'pinn' in results:
            record['pinn'] = results['pinn']['mean_error_percent']

        records.append(record)

    if not records:
        ax.text(0.5, 0.5, 'Insufficient data for noise sensitivity plot',
                ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        return

    # Sort records by noise level
    records_sorted = sorted(records, key=lambda x: x['noise'])

    # Extract noise levels
    noise_levels = [r['noise'] for r in records_sorted]

    # Helper to extract method errors, handling missing entries
    def get_method_errors(method_key):
        errors = []
        valid_noises = []
        for r in records_sorted:
            val = r.get(method_key, np.nan)
            if not np.isnan(val):
                errors.append(val)
                valid_noises.append(r['noise'])
        return valid_noises, errors

    ls_noises, ls_errors = get_method_errors('ls')
    nn_noises, nn_errors = get_method_errors('nn')
    attn_noises, attn_errors = get_method_errors('attn')
    pinn_noises, pinn_errors = get_method_errors('pinn')

    # Plot each method where data exists
    if ls_errors:
        ax.plot(ls_noises, ls_errors, 'o-', label='Least Squares', color='#2ecc71', markersize=8)
    if nn_errors:
        ax.plot(nn_noises, nn_errors, 's-', label='Neural Network', color='#3498db', markersize=8)
    if attn_errors:
        ax.plot(attn_noises, attn_errors, 'D-', label='Attention Network', color='#9b59b6', markersize=8)
    if pinn_errors:
        ax.plot(pinn_noises, pinn_errors, '^-', label='PINN', color='#e74c3c', markersize=8)

    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Mean Parameter Error (%)')
    ax.set_title('Parameter Identification Error vs Measurement Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_weights(all_results, save_path='attention_weights.png'):
    """
    Plot average attention weights across features for each run that has them.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    runs_with_attention = []
    attn_data = []  # list of matrices
    run_names = []
    
    for run_name, results in all_results.items():
        if results is None:
            continue
        if 'attention_network' in results and 'attention_weights' in results['attention_network']:
            weights = results['attention_network']['attention_weights']
            if weights is None:
                continue
            arr = np.array(weights)
            if arr.ndim >= 2:
                # If more dimensions, average over heads? Let's assume shape (heads,4,4) or (4,4)
                # For simplicity, take mean over heads dimension if present
                if arr.ndim == 3:
                    arr = arr.mean(axis=0)  # (4,4)
                elif arr.ndim == 4:
                    # (batch?, heads,4,4) unlikely
                    # average over batch and heads? We'll just take mean over first two dims
                    arr = arr.mean(axis=0).mean(axis=0)
                # Ensure it's 2D matrix
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    runs_with_attention.append(run_name)
                    attn_data.append(arr)
                    run_names.append(run_name)
    
    if not runs_with_attention:
        print("No attention weights found to plot")
        return
    
    # Determine grid layout
    n = len(runs_with_attention)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes_flat = axes.ravel()
    
    # Feature names for labeling
    feature_names = ['delta', 'v', 'beta', 'r']
    
    for idx, (run_name, matrix) in enumerate(zip(run_names, attn_data)):
        ax = axes_flat[idx]
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names)
        ax.set_yticklabels(feature_names)
        ax.set_title(f'{run_name}')
        plt.colorbar(im, ax=ax)
        # Annotate values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', color='w' if matrix[i,j] > 0.5 else 'black')
    
    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle('Attention Weights Across Features per Run')
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

    # Define descriptive labels for each run we want to include
    labels = {
        'run_0': 'Baseline (noise 0.01)',
        'run_1': 'Attention (noise 0.01)',
        'run_2': 'Attention (noise 0.02)',
        'run_3': 'Attention (noise 0.05)',
        'run_4': 'Attention-tuned (noise 0.05)',
        'run_5': 'Attention-tuned (noise 0.02)',
    }

    # Filter and rename runs
    filtered_results = {}
    for run_dir, results in all_results.items():
        if run_dir in labels:
            filtered_results[labels[run_dir]] = results
    all_results = filtered_results

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
