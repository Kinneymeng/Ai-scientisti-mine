"""
Vehicle Steering Parameter Identification using Machine Learning

This experiment uses a two-degree-of-freedom bicycle model to generate vehicle dynamics data,
then applies machine learning methods to identify key steering parameters (cornering stiffness).

The bicycle model equations:
    m * v * (beta_dot + r) = Fyf + Fyr
    Iz * r_dot = Lf * Fyf - Lr * Fyr

Where:
    - beta: sideslip angle
    - r: yaw rate
    - Fyf, Fyr: front/rear lateral tire forces
    - Cf, Cr: front/rear cornering stiffness (parameters to identify)
"""

import argparse
import json
import os
import numpy as np
from scipy.integrate import odeint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============== Vehicle Model Parameters ==============
class VehicleParams:
    """True vehicle parameters (to be identified)"""
    def __init__(self):
        self.m = 1500.0       # Vehicle mass [kg]
        self.Iz = 2500.0      # Yaw moment of inertia [kg*m^2]
        self.Lf = 1.2         # Distance from CG to front axle [m]
        self.Lr = 1.4         # Distance from CG to rear axle [m]
        self.Cf = 80000.0     # Front cornering stiffness [N/rad] - TO IDENTIFY
        self.Cr = 90000.0     # Rear cornering stiffness [N/rad] - TO IDENTIFY


# ============== Bicycle Model Simulation ==============
def bicycle_model(state, t, delta, v, params):
    """
    Two-DOF bicycle model dynamics

    State: [beta, r] - sideslip angle and yaw rate
    Input: delta - front wheel steering angle, v - vehicle speed
    """
    beta, r = state

    # Tire slip angles
    alpha_f = delta - beta - params.Lf * r / v
    alpha_r = -beta + params.Lr * r / v

    # Lateral tire forces (linear tire model)
    Fyf = params.Cf * alpha_f
    Fyr = params.Cr * alpha_r

    # State derivatives
    beta_dot = (Fyf + Fyr) / (params.m * v) - r
    r_dot = (params.Lf * Fyf - params.Lr * Fyr) / params.Iz

    return [beta_dot, r_dot]


def generate_simulation_data(params, num_samples=5000, noise_level=0.01, seed=42):
    """
    Generate vehicle dynamics data from bicycle model simulation
    """
    np.random.seed(seed)

    # Time settings
    dt = 0.01  # 100 Hz sampling

    # Storage for data
    data = {
        'delta': [],      # Steering angle input
        'velocity': [],   # Vehicle speed
        'beta': [],       # Sideslip angle
        'yaw_rate': [],   # Yaw rate
        'ay': [],         # Lateral acceleration
    }

    samples_collected = 0

    while samples_collected < num_samples:
        # Random initial conditions and inputs
        v = np.random.uniform(10, 30)  # Speed: 10-30 m/s
        delta_amplitude = np.random.uniform(0.01, 0.05)  # Steering amplitude
        freq = np.random.uniform(0.5, 2.0)  # Steering frequency

        # Simulation time for this maneuver
        t_sim = np.random.uniform(2, 5)
        t = np.arange(0, t_sim, dt)

        # Steering input (sinusoidal)
        delta = delta_amplitude * np.sin(2 * np.pi * freq * t)

        # Initial state
        state0 = [0.0, 0.0]

        # Simulate
        states = np.zeros((len(t), 2))
        states[0] = state0

        for i in range(1, len(t)):
            state_dot = bicycle_model(states[i-1], t[i-1], delta[i-1], v, params)
            states[i] = states[i-1] + np.array(state_dot) * dt

        beta = states[:, 0]
        r = states[:, 1]

        # Lateral acceleration
        ay = v * (np.gradient(beta, dt) + r)

        # Add measurement noise
        beta_noisy = beta + np.random.normal(0, noise_level * np.std(beta), len(beta))
        r_noisy = r + np.random.normal(0, noise_level * np.std(r), len(r))
        ay_noisy = ay + np.random.normal(0, noise_level * np.std(ay), len(ay))

        # Store data
        for i in range(len(t)):
            if samples_collected >= num_samples:
                break
            data['delta'].append(delta[i])
            data['velocity'].append(v)
            data['beta'].append(beta_noisy[i])
            data['yaw_rate'].append(r_noisy[i])
            data['ay'].append(ay_noisy[i])
            samples_collected += 1

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    return data


# ============== Neural Network Model ==============
class ParameterIdentificationNet(nn.Module):
    """
    Neural network for parameter identification
    Takes vehicle state and inputs, predicts cornering stiffness
    """
    def __init__(self, hidden_sizes=[64, 64], activation='relu'):
        super().__init__()

        input_size = 4  # delta, v, beta, r
        output_size = 2  # Cf, Cr estimates

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softplus())  # Ensure positive outputs

        self.network = nn.Sequential(*layers)

        # Scale factor for output (parameters are in 10^4 range)
        self.scale = 10000.0

    def forward(self, x):
        return self.network(x) * self.scale


class PhysicsInformedNet(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for parameter identification
    Embeds vehicle dynamics equations as constraints
    """
    def __init__(self, hidden_sizes=[64, 64]):
        super().__init__()

        # Network to estimate parameters
        input_size = 4
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 2))
        layers.append(nn.Softplus())

        self.param_net = nn.Sequential(*layers)
        self.scale = 10000.0

    def forward(self, x):
        return self.param_net(x) * self.scale

    def physics_loss(self, x, beta_dot, r_dot, params_known):
        """
        Compute physics-based loss using vehicle dynamics equations
        """
        delta, v, beta, r = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # Predicted parameters
        pred_params = self.forward(x)
        Cf_pred = pred_params[:, 0]
        Cr_pred = pred_params[:, 1]

        # Known vehicle parameters
        m, Iz, Lf, Lr = params_known

        # Tire slip angles
        alpha_f = delta - beta - Lf * r / v
        alpha_r = -beta + Lr * r / v

        # Predicted tire forces
        Fyf_pred = Cf_pred * alpha_f
        Fyr_pred = Cr_pred * alpha_r

        # Predicted state derivatives from physics
        beta_dot_pred = (Fyf_pred + Fyr_pred) / (m * v) - r
        r_dot_pred = (Lf * Fyf_pred - Lr * Fyr_pred) / Iz

        # Physics residual loss
        loss = torch.mean((beta_dot_pred - beta_dot)**2 + (r_dot_pred - r_dot)**2)

        return loss


# ============== Training Functions ==============
def train_standard_nn(model, train_loader, val_loader, true_params, epochs=100, lr=0.001, 
                      use_curriculum=False, start_noise=0.001, target_noise=0.01, 
                      data_generator=None, batch_size=64):
    """Train standard neural network with supervised learning
    
    Args:
        use_curriculum: If True, progressively increase noise during training
        start_noise: Starting noise level for curriculum learning
        target_noise: Target noise level for curriculum learning
        data_generator: Function to generate data with specified noise level
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Target: true cornering stiffness values
    target = torch.tensor([[true_params.Cf, true_params.Cr]], dtype=torch.float32)

    train_losses = []
    val_losses = []
    noise_schedule = []

    for epoch in range(epochs):
        # Calculate current noise level for curriculum learning
        if use_curriculum:
            progress = epoch / (epochs - 1) if epochs > 1 else 1.0
            current_noise = start_noise + (target_noise - start_noise) * progress
            noise_schedule.append(current_noise)
            
            # Regenerate training data with current noise level
            if data_generator is not None:
                curriculum_data = data_generator(true_params, num_samples=int(0.8 * 5000), 
                                                 noise_level=current_noise, seed=42 + epoch)
                x_train = torch.tensor(np.column_stack([
                    curriculum_data['delta'], curriculum_data['velocity'], 
                    curriculum_data['beta'], curriculum_data['yaw_rate']
                ]), dtype=torch.float32)
                y_train = torch.tensor(curriculum_data['ay'], dtype=torch.float32).unsqueeze(1)
                train_dataset = TensorDataset(x_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        epoch_loss = 0

        for batch_x, _ in train_loader:
            optimizer.zero_grad()

            pred_params = model(batch_x)
            # Loss: how close predictions are to true parameters
            loss = criterion(pred_params, target.expand(batch_x.size(0), -1))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_x, _ in val_loader:
                pred_params = model(batch_x)
                loss = criterion(pred_params, target.expand(batch_x.size(0), -1))
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

    return train_losses, val_losses, noise_schedule


def train_pinn(model, data, true_params, epochs=100, lr=0.001):
    """Train Physics-Informed Neural Network"""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare data
    x = torch.tensor(np.column_stack([
        data['delta'], data['velocity'], data['beta'], data['yaw_rate']
    ]), dtype=torch.float32)

    # Compute state derivatives from data
    dt = 0.01
    beta_dot = torch.tensor(np.gradient(data['beta'], dt), dtype=torch.float32)
    r_dot = torch.tensor(np.gradient(data['yaw_rate'], dt), dtype=torch.float32)

    params_known = (true_params.m, true_params.Iz, true_params.Lf, true_params.Lr)

    losses = []
    param_errors = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Physics-informed loss
        loss = model.physics_loss(x, beta_dot, r_dot, params_known)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Track parameter estimation error
        with torch.no_grad():
            pred_params = model(x).mean(dim=0)
            cf_error = abs(pred_params[0].item() - true_params.Cf) / true_params.Cf * 100
            cr_error = abs(pred_params[1].item() - true_params.Cr) / true_params.Cr * 100
            param_errors.append((cf_error + cr_error) / 2)

    return losses, param_errors


# ============== Baseline: Least Squares ==============
def least_squares_identification(data, true_params):
    """
    Classical least squares parameter identification
    """
    # Build regressor matrix for linear tire model
    delta = data['delta']
    v = data['velocity']
    beta = data['beta']
    r = data['yaw_rate']

    dt = 0.01
    beta_dot = np.gradient(beta, dt)
    r_dot = np.gradient(r, dt)

    m, Iz, Lf, Lr = true_params.m, true_params.Iz, true_params.Lf, true_params.Lr

    # Tire slip angles
    alpha_f = delta - beta - Lf * r / v
    alpha_r = -beta + Lr * r / v

    # Linear regression: y = A * [Cf, Cr]^T
    # From beta_dot equation: m*v*(beta_dot + r) = Cf*alpha_f + Cr*alpha_r
    y1 = m * v * (beta_dot + r)
    A1 = np.column_stack([alpha_f, alpha_r])

    # From r_dot equation: Iz*r_dot = Lf*Cf*alpha_f - Lr*Cr*alpha_r
    y2 = Iz * r_dot
    A2 = np.column_stack([Lf * alpha_f, -Lr * alpha_r])

    # Combine
    y = np.concatenate([y1, y2])
    A = np.vstack([A1, A2])

    # Least squares solution
    params_est, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    Cf_est = params_est[0]
    Cr_est = params_est[1]

    return Cf_est, Cr_est


# ============== Evaluation ==============
def evaluate_identification(pred_Cf, pred_Cr, true_params):
    """Calculate identification errors"""
    cf_error = abs(pred_Cf - true_params.Cf) / true_params.Cf * 100
    cr_error = abs(pred_Cr - true_params.Cr) / true_params.Cr * 100
    mean_error = (cf_error + cr_error) / 2

    return {
        'cf_error_percent': cf_error,
        'cr_error_percent': cr_error,
        'mean_error_percent': mean_error,
        'pred_Cf': pred_Cf,
        'pred_Cr': pred_Cr,
        'true_Cf': true_params.Cf,
        'true_Cr': true_params.Cr,
    }


# ============== Main Experiment ==============
def run_experiment(
    seed=42,
    num_samples=5000,
    noise_level=0.01,
    hidden_sizes=[64, 64],
    activation='relu',
    epochs=100,
    learning_rate=0.001,
    batch_size=64,
    use_pinn=False,
    use_curriculum=False,
    start_noise=0.001,
    test_noise_levels=None,
):
    """
    Run the parameter identification experiment
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize true parameters
    true_params = VehicleParams()

    # Generate simulation data
    print("Generating simulation data...")
    data = generate_simulation_data(true_params, num_samples, noise_level, seed)

    # Prepare PyTorch datasets
    x = torch.tensor(np.column_stack([
        data['delta'], data['velocity'], data['beta'], data['yaw_rate']
    ]), dtype=torch.float32)
    y = torch.tensor(data['ay'], dtype=torch.float32).unsqueeze(1)

    # Train/validation split
    split_idx = int(0.8 * len(x))
    train_dataset = TensorDataset(x[:split_idx], y[:split_idx])
    val_dataset = TensorDataset(x[split_idx:], y[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    results = {}

    # Baseline: Least Squares
    print("Running least squares identification...")
    ls_Cf, ls_Cr = least_squares_identification(data, true_params)
    results['least_squares'] = evaluate_identification(ls_Cf, ls_Cr, true_params)
    print(f"  Cf error: {results['least_squares']['cf_error_percent']:.2f}%")
    print(f"  Cr error: {results['least_squares']['cr_error_percent']:.2f}%")

    # Neural Network
    print("Training neural network...")
    model = ParameterIdentificationNet(hidden_sizes, activation)
    train_losses, val_losses, noise_schedule = train_standard_nn(
        model, train_loader, val_loader, true_params, epochs, learning_rate,
        use_curriculum=use_curriculum, start_noise=start_noise, 
        target_noise=noise_level, data_generator=generate_simulation_data,
        batch_size=batch_size
    )

    # Get final predictions
    model.eval()
    with torch.no_grad():
        all_preds = model(x)
        nn_Cf = all_preds[:, 0].mean().item()
        nn_Cr = all_preds[:, 1].mean().item()

    results['neural_network'] = evaluate_identification(nn_Cf, nn_Cr, true_params)
    results['neural_network']['train_losses'] = train_losses
    results['neural_network']['val_losses'] = val_losses
    results['neural_network']['noise_schedule'] = noise_schedule
    print(f"  Cf error: {results['neural_network']['cf_error_percent']:.2f}%")
    print(f"  Cr error: {results['neural_network']['cr_error_percent']:.2f}%")

    # Physics-Informed Neural Network
    if use_pinn:
        print("Training physics-informed neural network...")
        pinn_model = PhysicsInformedNet(hidden_sizes)
        pinn_losses, pinn_param_errors = train_pinn(
            pinn_model, data, true_params, epochs, learning_rate
        )

        pinn_model.eval()
        with torch.no_grad():
            pinn_preds = pinn_model(x)
            pinn_Cf = pinn_preds[:, 0].mean().item()
            pinn_Cr = pinn_preds[:, 1].mean().item()

        results['pinn'] = evaluate_identification(pinn_Cf, pinn_Cr, true_params)
        results['pinn']['losses'] = pinn_losses
        results['pinn']['param_errors'] = pinn_param_errors
        print(f"  Cf error: {results['pinn']['cf_error_percent']:.2f}%")
        print(f"  Cr error: {results['pinn']['cr_error_percent']:.2f}%")

    # Summary metrics
    results['summary'] = {
        'noise_level': noise_level,
        'num_samples': num_samples,
        'epochs': epochs,
        'use_curriculum': use_curriculum,
        'best_method': min(
            ['least_squares', 'neural_network'] + (['pinn'] if use_pinn else []),
            key=lambda m: results[m]['mean_error_percent']
        ),
    }

    # Generalization test: evaluate on unseen noise levels
    if test_noise_levels is not None:
        print("\nRunning generalization tests on unseen noise levels...")
        results['generalization'] = {}
        
        for test_noise in test_noise_levels:
            print(f"  Testing on noise level: {test_noise}")
            test_data = generate_simulation_data(true_params, num_samples=1000, 
                                                  noise_level=test_noise, seed=seed+100)
            x_test = torch.tensor(np.column_stack([
                test_data['delta'], test_data['velocity'], 
                test_data['beta'], test_data['yaw_rate']
            ]), dtype=torch.float32)
            
            model.eval()
            with torch.no_grad():
                test_preds = model(x_test)
                test_Cf = test_preds[:, 0].mean().item()
                test_Cr = test_preds[:, 1].mean().item()
            
            gen_results = evaluate_identification(test_Cf, test_Cr, true_params)
            results['generalization'][f'noise_{test_noise}'] = {
                'cf_error_percent': gen_results['cf_error_percent'],
                'cr_error_percent': gen_results['cr_error_percent'],
                'mean_error_percent': gen_results['mean_error_percent'],
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='Vehicle Steering Parameter Identification')
    parser.add_argument('--out_dir', type=str, default='run_0', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of data samples')
    parser.add_argument('--noise_level', type=float, default=0.01, help='Measurement noise level')
    parser.add_argument('--hidden_sizes', type=str, default='64,64', help='Hidden layer sizes')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'leaky_relu'])
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--use_pinn', action='store_true', help='Use Physics-Informed NN')
    parser.add_argument('--use_curriculum', action='store_true', help='Use curriculum learning with progressive noise')
    parser.add_argument('--start_noise', type=float, default=0.001, help='Starting noise level for curriculum learning')
    parser.add_argument('--test_noise_levels', type=str, default=None, help='Comma-separated noise levels for generalization testing')

    args = parser.parse_args()

    # Parse hidden sizes
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]

    # Parse test noise levels for generalization
    test_noise_levels = None
    if args.test_noise_levels:
        test_noise_levels = [float(x) for x in args.test_noise_levels.split(',')]

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Run experiment
    results = run_experiment(
        seed=args.seed,
        num_samples=args.num_samples,
        noise_level=args.noise_level,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_pinn=args.use_pinn,
        use_curriculum=args.use_curriculum,
        start_noise=args.start_noise,
        test_noise_levels=test_noise_levels,
    )

    # Save detailed results
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    results_json = convert_for_json(results)

    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save final_info.json in required format
    final_info = {
        'ls_cf_error': {
            'means': results['least_squares']['cf_error_percent'],
            'stds': 0.0,
        },
        'ls_cr_error': {
            'means': results['least_squares']['cr_error_percent'],
            'stds': 0.0,
        },
        'nn_cf_error': {
            'means': results['neural_network']['cf_error_percent'],
            'stds': 0.0,
        },
        'nn_cr_error': {
            'means': results['neural_network']['cr_error_percent'],
            'stds': 0.0,
        },
        'nn_mean_error': {
            'means': results['neural_network']['mean_error_percent'],
            'stds': 0.0,
        },
        'best_method': {
            'means': 0 if results['summary']['best_method'] == 'neural_network' else 1,
            'stds': 0.0,
        },
    }

    if 'pinn' in results:
        final_info['pinn_cf_error'] = {
            'means': results['pinn']['cf_error_percent'],
            'stds': 0.0,
        }
        final_info['pinn_cr_error'] = {
            'means': results['pinn']['cr_error_percent'],
            'stds': 0.0,
        }
        final_info['pinn_mean_error'] = {
            'means': results['pinn']['mean_error_percent'],
            'stds': 0.0,
        }

    with open(os.path.join(args.out_dir, 'final_info.json'), 'w') as f:
        json.dump(final_info, f, indent=2)

    print(f"\nResults saved to {args.out_dir}/")
    print(f"Best method: {results['summary']['best_method']}")

    return results


if __name__ == '__main__':
    main()
