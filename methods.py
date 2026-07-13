import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import gaussian_kde

class GammaMLP(nn.Module):
    """Neural Network to parameterize the limit cycle smoothly and periodically."""
    def __init__(self, d_out, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d_out)
        )
    def forward(self, cos_sin):
        return self.net(cos_sin)


class SpaceToCircleMLP(nn.Module):
    """Neural Network to smoothly map ambient space R^d to S^1 coordinates (cos, sin)."""
    def __init__(self, d_in, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        out = self.net(x)
        # Normalize to ensure outputs lie strictly on the unit circle
        return out / (torch.norm(out, dim=1, keepdim=True) + 1e-8)


class LimitCycleVectorField:
    def __init__(self, d_space, hidden_dim=64):
        self.d_space = d_space
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Both networks are now smooth MLPs
        self.gamma_net = GammaMLP(d_out=d_space, hidden_dim=32).to(self.device)
        self.coord_net = SpaceToCircleMLP(d_in=d_space, hidden_dim=hidden_dim).to(self.device)
        
        self.kde = None
        self.pca_to_umap = None

    def fit_gamma(self, theta, X_pca, epochs=500, lr=1e-3, batch_size=64, smooth_penalty=0.1):
        """Step 1: Fit gamma(theta) with an explicit smoothness regularizer."""
        self.gamma_net.train()
        optimizer = optim.Adam(self.gamma_net.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        tensor_input = torch.tensor(np.stack([np.cos(theta), np.sin(theta)], axis=1), dtype=torch.float32).to(self.device)
        tensor_target = torch.tensor(X_pca, dtype=torch.float32).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(tensor_input, tensor_target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        theta_grid = torch.linspace(0, 2 * np.pi, 100).to(self.device)
        cos_sin_grid = torch.stack([torch.cos(theta_grid), torch.sin(theta_grid)], dim=1)
        
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}")
            for batch_in, batch_tar in loader:
                optimizer.zero_grad()
                pred = self.gamma_net(batch_in)
                mse_loss = criterion(pred, batch_tar)
                
                gamma_grid = self.gamma_net(cos_sin_grid)
                diffs = gamma_grid[1:] - gamma_grid[:-1]
                L_smooth = torch.mean(diffs ** 2)
                
                loss = mse_loss + smooth_penalty * L_smooth
                loss.backward()
                optimizer.step()
                
        self.gamma_net.eval()

    def fit_density(self, theta):
        """Step 2: Estimate 1D density using a wrapped boundary approach."""
        theta_extended = np.concatenate([theta - 2 * np.pi, theta, theta + 2 * np.pi])
        self.kde = gaussian_kde(theta_extended)

    def fit_coordinate_net(self, X_pca, theta, epochs=500, lr=1e-3, batch_size=64):
        """Step 4 Replacement: Train a smooth MLP to map R^d -> S^1 coordinates."""
        self.coord_net.train()
        optimizer = optim.Adam(self.coord_net.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        target_cos_sin = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        
        tensor_input = torch.tensor(X_pca, dtype=torch.float32).to(self.device)
        tensor_target = torch.tensor(target_cos_sin, dtype=torch.float32).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(tensor_input, tensor_target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_in, batch_tar in loader:
                optimizer.zero_grad()
                pred = self.coord_net(batch_in)
                loss = criterion(pred, batch_tar)
                loss.backward()
                optimizer.step()
                
        self.coord_net.eval()

    def fit_umap_projector(self, X_pca, X_umap):
        """Fit a geometric mapping helper from PCA to UMAP for visualization."""
        from sklearn.neighbors import KNeighborsRegressor
        self.pca_to_umap = KNeighborsRegressor(n_neighbors=15, weights='distance')
        self.pca_to_umap.fit(X_pca, X_umap)

    def predict_theta(self, X):
        """Predict theta smoothly for arbitrary points using the trained Coordinate Net."""
        self.coord_net.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            cos_sin = self.coord_net(X_tensor).cpu().numpy()
        theta = np.arctan2(cos_sin[:, 1], cos_sin[:, 0])
        return np.mod(theta, 2 * np.pi)

    def _get_gamma_and_tangent(self, theta_array):
        """Use PyTorch autograd to evaluate exact gamma positions and normalized tangents."""
        theta_tensor = torch.tensor(theta_array, dtype=torch.float32, requires_grad=True).to(self.device)
        cos_sin = torch.stack([torch.cos(theta_tensor), torch.sin(theta_tensor)], dim=1)
        gamma = self.gamma_net(cos_sin)
        
        tangents = []
        for i in range(self.d_space):
            grad_outputs = torch.zeros_like(gamma)
            grad_outputs[:, i] = 1.0
            grad = torch.autograd.grad(outputs=gamma, inputs=theta_tensor, 
                                       grad_outputs=grad_outputs, 
                                       retain_graph=True, only_inputs=True)[0]
            tangents.append(grad)
            
        tangents = torch.stack(tangents, dim=1)
        gamma_np = gamma.detach().cpu().numpy()
        tangents_np = tangents.detach().cpu().numpy()
        
        norms = np.linalg.norm(tangents_np, axis=1, keepdims=True) + 1e-8
        unit_tangents = tangents_np / norms
        return gamma_np, unit_tangents

    def compute_velocity(self, X, C=1.0, lmbda=0.1):
        """Steps 2, 3, & 4: Compute the global smooth vector field v(x)."""
        theta = self.predict_theta(X)
        gamma, unit_tangents = self._get_gamma_and_tangent(theta)
        
        rho = self.kde(theta) + 1e-5
        v_parallel = (C / rho)[:, np.newaxis] * unit_tangents
        v_perpendicular = -lmbda * (X - gamma)
        
        return v_parallel + v_perpendicular

    def simulate_trajectory(self, x0, steps=200, dt=0.02, C=1.0, lmbda=0.1):
        """Integrate an arbitrary starting position forward through the vector field."""
        trajectory = [x0]
        current_x = np.array(x0, dtype=np.float32).reshape(1, -1)
        
        for _ in range(steps):
            v = self.compute_velocity(current_x, C=C, lmbda=lmbda)
            current_x = current_x + dt * v
            trajectory.append(current_x.flatten())
            
        return np.array(trajectory)

    def project_velocity_to_umap(self, X_pca, velocities, dt=1e-3):
        if self.pca_to_umap is None:
            raise ValueError("UMAP projector has not been fitted.")
        X_umap = self.pca_to_umap.predict(X_pca)
        X_next_pca = X_pca + dt * velocities
        X_next_umap = self.pca_to_umap.predict(X_next_pca)
        v_umap = (X_next_umap - X_umap) / dt
        return X_umap, v_umap