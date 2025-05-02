import torch
import numpy as np

def add_noise(images, noise_factor=0.5):
	noisy = images+torch.randn_like(images) * noise_factor
	noisy = torch.clip(noisy,0.,1.)
	return noisy

def add_gaussian_noise(images, sigma=0.5):
    noise = torch.randn_like(images) * sigma
    return torch.clamp(images + noise, 0., 1.)

