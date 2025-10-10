import torch
import torch.nn.functional as F

def add_poisson_noise(image, scale_factor=1000):
    """
    Add realistic Poisson (shot) noise to an image
    
    Args:
        image: Input tensor in range [0,1]
        scale_factor: Higher values = less noise (more photons)
    """
    # Convert to photon counts (scale up to simulate photon detection)
    photon_counts = image * scale_factor
    
    # Apply Poisson noise 
    noisy_photons = torch.poisson(photon_counts)
    
    # Convert back to [0,1] range
    noisy_image = noisy_photons / scale_factor
    
    return torch.clamp(noisy_image, 0, 1)

def convolution(image, kernel):    
    if kernel.size(2) % 2 == 0 or kernel.size(3) % 2 == 0:
        raise ValueError("Kernel size must be odd")
    # Apply convolution with padding to maintain image size
    padding = (kernel.size(2) // 2, kernel.size(3) // 2)
    filtered = F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
    
    # Remove batch and channel dimensions: [1, 1, H, W] -> [H, W]
    return filtered.squeeze(0).squeeze(0)

def deconvolution(blurred_image, kernel, epsilon=1e-5):
    # Compute Fourier transforms
    P_blurred = torch.fft.fft2(blurred_image)
    B = torch.fft.fft2(kernel.squeeze(), s=blurred_image.shape)
    
    # Avoid division by zero by adding a small constant (epsilon)
    B_conj = torch.conj(B)
    B_magnitude_squared = torch.abs(B)**2
    B_inv = B_conj / (B_magnitude_squared + epsilon)
    
    # Perform deconvolution in the frequency domain
    P_deblurred = P_blurred * B_inv

    # Inverse Fourier transform to get the deblurred image
    deblurred_image = torch.fft.ifft2(P_deblurred).real
    
    return torch.clamp(deblurred_image, 0, 1)

def gaussian_normalised_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    ax = torch.arange(-size + 1, size + 1)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, size, size]