"""
Core diffusion model implementation for 3D brain volumes.

This module implements the denoising diffusion probabilistic model (DDPM)
and the denoising diffusion implicit model (DDIM) for 3D brain MRI data.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

class DiffusionModel:
    """Core implementation of diffusion model for 3D brain volumes."""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=None):
        """Initialize diffusion process parameters.
        
        Args:
            num_timesteps: Number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
            device: Computation device (CPU/GPU)
        """
        self.num_timesteps = num_timesteps
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define noise schedule (linear or cosine)
        self.betas = self._linear_beta_schedule(beta_start, beta_end)
        
        # Compute diffusion process constants
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def _linear_beta_schedule(self, beta_start, beta_end):
        """Linear noise schedule."""
        return torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device)
    
    def _cosine_beta_schedule(self, s=0.008):
        """Cosine noise schedule for improved sample quality."""
        steps = self.num_timesteps + 1
        t = torch.linspace(0, self.num_timesteps, steps, device=self.device) / self.num_timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def forward_diffusion(self, x_0, t):
        """Forward diffusion process q(x_t | x_0).
        
        Args:
            x_0: Original clean images [B, C, D, H, W]
            t: Timesteps [B]
            
        Returns:
            x_t: Noisy images at timestep t
            noise: The noise added
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        
        # Mean + variance
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def sample_timesteps(self, n):
        """Sample timesteps uniformly for training."""
        return torch.randint(low=0, high=self.num_timesteps, size=(n,), device=self.device)
    
    @torch.no_grad()
    def sample_ddpm(self, model, shape, conditioning=None):
        """Traditional DDPM sampling (slow but high quality)."""
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        # Progressively denoise
        for t in tqdm(reversed(range(self.num_timesteps)), desc="DDPM Sampling"):
            # Create batch of same timestep
            timestep = torch.full((shape[0],), t, dtype=torch.long, device=self.device)
            
            # Get model prediction
            predicted_noise = model(x, timestep, conditioning)
            
            # Get alpha and beta values for timestep
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # No noise on last step
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # Update x sample with DDPM update
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        
        return x
    
    @torch.no_grad()
    def sample_ddim(self, model, shape, conditioning=None, n_steps=50, guidance_scale=3.0):
        """Sample using DDIM for faster generation.
        
        Args:
            model: Noise prediction model
            shape: Shape of samples to generate
            conditioning: Optional conditioning information
            n_steps: Number of DDIM steps (much less than DDPM)
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated samples
        """
        # Implementation of DDIM sampling
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        
        # Subset of timesteps for DDIM
        times = torch.linspace(0, self.num_timesteps - 1, n_steps, device=self.device).long()
        times = list(reversed(times.int().cpu().numpy()))
        
        for i, t in enumerate(tqdm(times, desc="DDIM Sampling")):
            # Get diffusion timestep
            time_tensor = torch.tensor([t] * batch_size, device=self.device)
            
            # Classifier-free guidance
            if guidance_scale > 1.0 and conditioning is not None:
                # Predict noise with and without conditioning
                with torch.no_grad():
                    noise_pred_cond = model(x, time_tensor, conditioning)
                    noise_pred_uncond = model(x, time_tensor, None)
                
                # Apply guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = model(x, time_tensor, conditioning)
            
            # DDIM update step
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[times[i+1]] if i < len(times)-1 else torch.tensor(1.0, device=self.device)
            
            # Compute x_0 prediction
            x_0_pred = (x - noise_pred * torch.sqrt(1 - alpha)) / torch.sqrt(alpha)
            
            # DDIM formula
            c1 = torch.sqrt(alpha_prev)
            c2 = torch.sqrt(1 - alpha_prev)
            x = c1 * x_0_pred + c2 * noise_pred
            
        return x


def diffusion_loss_fn(model, diffusion, x_0, t, conditioning=None):
    """Noise prediction loss function.
    
    Args:
        model: Noise prediction model
        diffusion: Diffusion model
        x_0: Original clean images
        t: Timesteps
        conditioning: Optional conditioning information
        
    Returns:
        Loss value
    """
    # Add noise according to timestep
    x_t, noise = diffusion.forward_diffusion(x_0, t)
    
    # Predict the noise
    predicted_noise = model(x_t, t, conditioning)
    
    # Simple MSE loss 
    loss = F.mse_loss(predicted_noise, noise)
    return loss
