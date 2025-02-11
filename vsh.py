import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from gsplat.cuda._wrapper import spherical_harmonics
def compute_real_sh(degree, directions):
    """
    Fixed version with proper orthonormalization.
    Coefficients follow the widely used real SH convention from:
    "Spherical Harmonic Lighting: The Gritty Details" (Robin Green)
    """
    x, y, z = directions.unbind(-1)
    sh = []

    # Degree 0
    l0 = 0.5 * np.sqrt(1.0 / np.pi) * torch.ones_like(x)
    sh.append(l0)

    if degree >= 1:
        # Degree 1
        l1_m1 = np.sqrt(3.0 / (4*np.pi)) * y  # Correct normalization
        l1_0  = np.sqrt(3.0 / (4*np.pi)) * z
        l1_p1 = np.sqrt(3.0 / (4*np.pi)) * x
        sh.extend([l1_m1, l1_0, l1_p1])

    if degree >= 2:
        # Degree 2
        xy = x * y
        yz = y * z
        xz = x * z
        x2 = x**2
        y2 = y**2
        z2 = z**2

        l2_m2 = 0.5 * np.sqrt(15.0 / np.pi) * xy
        l2_m1 = 0.5 * np.sqrt(15.0 / np.pi) * yz
        l2_0  = 0.25 * np.sqrt(5.0 / np.pi) * (3*z2 - 1)
        l2_p1 = 0.5 * np.sqrt(15.0 / np.pi) * xz
        l2_p2 = 0.25 * np.sqrt(15.0 / np.pi) * (x2 - y2)
        sh.extend([l2_m2, l2_m1, l2_0, l2_p1, l2_p2])

    return torch.stack(sh, dim=-1)
class SHDistribution(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.num_coeffs = (degree + 1)**2  # Number of SH coefficients
        self.coeffs = nn.Parameter(torch.randn(self.num_coeffs)).cuda()  # Trainable SH coefficients

    def forward(self, directions):
        # Ensure input directions are unit vectors
        directions = F.normalize(directions, p=2, dim=-1)

        coeffs = self.coeffs.unsqueeze(0).repeat(directions.size(0), 1)


        # adapt for gsplat version
        # coeffs = coeffs.reshape(directions.shape[0], self.num_coeffs, 3)
        coeffs = coeffs.unsqueeze(-1).repeat(1, 1, 3)
        
        # # Compute SH basis values for the directions
        # sh_basis = compute_real_sh(self.degree, directions)  # (batch_size, num_coeffs)
        
        # # Compute linear combination of SH coefficients
        # f1 = torch.einsum('bi,i->b', sh_basis, self.coeffs)  # (batch_size,)


        f2 = spherical_harmonics(self.degree, directions.cuda(), coeffs.cuda())[:, 0]
        
        # Compute squared value and normalize
        pdf_unnorm = f2**2
        norm = torch.sum(self.coeffs**2)  # Integral of f^2 over the sphere
        pdf = pdf_unnorm / (norm + 1e-10)  # Avoid division by zero
        
        return pdf


# Initialize model (degree=2 uses 9 coefficients)
model = SHDistribution(degree=2)

# Example input directions (batch_size, 3)
directions = torch.randn(10, 3)
directions = F.normalize(directions, p=2, dim=-1)

# Compute PDF values
pdf = model(directions)
print("PDF:", pdf)
import torch
import torch.nn.functional as F

def verify_integral(model, n_samples=1_000_0000):
    # Generate random directions on the sphere
    directions = torch.randn(n_samples, 3).cuda()
    directions = F.normalize(directions, p=2, dim=-1)
    
    # Compute PDF values
    with torch.no_grad():
        pdf = model(directions)
    
    # Monte Carlo integration: (4π * average(pdf))
    integral = (4 * torch.pi) * (pdf.sum() / n_samples)
    return integral.item()

# Test with different coefficient configurations
if __name__ == "__main__":
    # Case 1: Default random coefficients
    model = SHDistribution(degree=2)
    print(f"Random coefficients integral: {verify_integral(model):.6f}")

    # Case 2: Only the constant term (degree 0)
    with torch.no_grad():
        model.coeffs[1:].zero_()  # Zero out all non-constant terms
    print(f"Only degree 0 integral: {verify_integral(model):.6f}")

    # Case 3: Specific known configuration
    with torch.no_grad():
        model.coeffs[:] = 0
        model.coeffs[0] = 1.0  # Set constant term
    print(f"Unit constant term integral: {verify_integral(model):.6f}")