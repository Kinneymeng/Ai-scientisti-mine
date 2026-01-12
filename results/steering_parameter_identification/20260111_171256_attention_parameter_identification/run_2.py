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
import re
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


# ============== Attention-Enhanced Neural Network ==============
class AttentionParameterIdentificationNet(nn.Module):
    """
    Neural network with self-attention across input features.
    """
    def __init__(self, hidden_sizes=[64, 64], activation='relu', attention_heads=4):
        super().__init__()
        input_size = 4
        d_model = 64  # embedding dimension for each feature
        self.d_model = d_model
        self.num_features = input_size

        # Embed each scalar feature into d_model
        self.feature_embed = nn.Linear(1, d_model)
        # Multi‑head self‑attention across the four features
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attention_heads,
            batch_first=True
        )
        # MLP after mean‑pooling over features
        mlp_input_size = d_model
        layers = []
        prev_size = mlp_input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))
        layers.append(nn.Softplus())
        self.mlp = nn.Sequential(*layers)
        self.scale = 10000.0

    def forward(self, x):
        # x shape: (batch, 4)
        batch_size = x.size(0)
        # treat each feature as a separate token: (batch, 4, 1)
        x_reshaped = x.unsqueeze(-1)
        embedded = self.feature_embed(x_reshaped)   # (batch, 4, d_model)
        # self‑attention
        attn_output, attn_weights = self.attention(embedded, embedded, embedded)
        # mean pooling over the feature dimension
        pooled = attn_output.mean(dim=1)            # (batch, d_model)
        out = self.mlp(pooled) * self.scale
        return out, attn_weights


# ============== Physics-Informed Neural Network ==============
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
def train_standard_nn(model, train_loader, val_loader, true_params, epochs=100, lr=0.001):
    """Train standard neural network with supervised learning"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Target: true cornering stiffness values
    target = torch.tensor([[true_params.Cf, true_params.Cr]], dtype=torch.float32)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
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

    return train_losses, val_losses


def train_attention_nn(model, train_loader, val_loader, true_params, epochs=100, lr=0.001):
    """Train attention‑enhanced neural network with supervised learning"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    target = torch.tensor([[true_params.Cf, true_params.Cr]], dtype=torch.float32)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            pred_params, _ = model(batch_x)
            loss = criterion(pred_params, target.expand(batch_x.size(0), -1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_x, _ in val_loader:
                pred_params, _ = model(batch_x)
                loss = criterion(pred_params, target.expand(batch_x.size(0), -1))
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

    return train_losses, val_losses


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
    train_losses, val_losses = train_standard_nn(
        model, train_loader, val_loader, true_params, epochs, learning_rate
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
    print(f"  Cf error: {results['neural_network']['cf_error_percent']:.2f}%")
    print(f"  Cr error: {results['neural_network']['cr_error_percent']:.2f}%")

    # Attention‑Enhanced Neural Network
    print("Training attention‑enhanced neural network...")
    attention_model = AttentionParameterIdentificationNet(hidden_sizes, activation, attention_heads=4)
    att_train_losses, att_val_losses = train_attention_nn(
        attention_model, train_loader, val_loader, true_params, epochs, learning_rate
    )
    attention_model.eval()
    with torch.no_grad():
        all_preds_att = []
        all_attn_weights = []
        for batch_x, _ in val_loader:
            pred_params, attn_weights = attention_model(batch_x)
            all_preds_att.append(pred_params)
            all_attn_weights.append(attn_weights.detach().cpu())
        # Concatenate predictions and compute mean Cf, Cr
        if all_preds_att:
            preds_cat = torch.cat(all_preds_att, dim=0)
            att_Cf = preds_cat[:, 0].mean().item()
            att_Cr = preds_cat[:, 1].mean().item()
        else:
            att_Cf, att_Cr = 0.0, 0.0
        # Compute average attention weights across validation set
        if all_attn_weights:
            stacked_attn = torch.cat(all_attn_weights, dim=0)  # (total_batch, ...)
            mean_attn = stacked_attn.mean(dim=0)               # average over batch
            attn_weights_np = mean_attn.numpy().tolist()
        else:
            attn_weights_np = None

    results['attention_network'] = evaluate_identification(att_Cf, att_Cr, true_params)
    results['attention_network']['train_losses'] = att_train_losses
    results['attention_network']['val_losses'] = att_val_losses
    results['attention_network']['attention_weights'] = attn_weights_np
    print(f"  Cf error: {results['attention_network']['cf_error_percent']:.2f}%")
    print(f"  Cr error: {results['attention_network']['cr_error_percent']:.2f}%")

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
        'best_method': min(
            ['least_squares', 'neural_network'] + (['pinn'] if use_pinn else []),
            key=lambda m: results[m]['mean_error_percent']
        ),
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

    args = parser.parse_args()

    # Determine noise level based on run number
    run_match = re.search(r'run_(\d+)', args.out_dir)
    if run_match:
        run_num = int(run_match.group(1))
        # Map run numbers to noise levels for attention experiments
        if run_num == 1:
            args.noise_level = 0.01
        elif run_num == 2:
            args.noise_level = 0.02
        elif run_num == 3:
            args.noise_level = 0.05
        # else keep default

    # Parse hidden sizes
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]

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
        'attention_cf_error': {
            'means': results['attention_network']['cf_error_percent'],
            'stds': 0.0,
        },
        'attention_cr_error': {
            'means': results['attention_network']['cr_error_percent'],
            'stds': 0.0,
        },
        'attention_mean_error': {
            'means': results['attention_network']['mean_error_percent'],
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
