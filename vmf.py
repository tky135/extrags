import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VonMisesFisherMixture(nn.Module):
    def __init__(self, num_components):
        super().__init__()
        self.num_components = num_components
        self.mixture_logits = nn.Parameter(torch.randn(num_components))
        self.mu = nn.Parameter(torch.randn(num_components, 3))  # Unnormalized means
        self.kappa_log = nn.Parameter(torch.zeros(num_components))  # Log concentration parameters

    def forward(self, directions):
        # Ensure input directions are unit vectors (assumed)
        directions = F.normalize(directions, p=2, dim=-1)  # (batch_size, 3)
        
        # Normalize mean vectors to unit length
        mu = F.normalize(self.mu, p=2, dim=1)  # (num_components, 3)
        
        # Compute concentration parameters (clamped for stability)
        kappa = torch.exp(self.kappa_log)  # (num_components,)
        kappa = torch.clamp(kappa, max=50)  # Prevents overflow
        
        # Compute dot products between directions and means
        dots = torch.einsum('bi,ki->bk', directions, mu)  # (batch_size, num_components)
        
        # Compute log normalization constants log(C)
        log_sinh_kappa = torch.log(torch.sinh(kappa))
        log_C = torch.log(kappa) - np.log(4 * np.pi) - log_sinh_kappa  # (num_components,)
        
        # Log densities for each component: log(C) + kappa * (mu^T direction)
        log_f = log_C[None, :] + kappa[None, :] * dots  # (batch_size, num_components)
        
        # Log mixture weights
        log_weights = F.log_softmax(self.mixture_logits, dim=0)  # (num_components,)
        
        # Combine and logsumexp over components
        log_pdf = torch.logsumexp(log_weights[None, :] + log_f, dim=1)  # (batch_size,)
        
        return torch.exp(log_pdf)

# Example usage
num_components = 5
model = VonMisesFisherMixture(num_components)

# Sample directions (batch_size, 3)
directions = torch.randn(10, 3)
directions = F.normalize(directions, p=2, dim=-1)

# Compute PDF
pdf = model(directions)  # (batch_size,)
print("PDF:", pdf)