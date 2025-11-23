import torch
import torch.nn.functional as F

def add_poisson_noise(image, scale_factor=1000):
    # Convert to photon counts (scale up to simulate photon detection)
    photon_counts = image * scale_factor
    
    # Apply Poisson noise 
    noisy_photons = torch.poisson(photon_counts)
    
    # Convert back to [0,1] range
    noisy_image = noisy_photons / scale_factor
    
    return torch.clamp(noisy_image, 0, 1)

def gaussian_normalised_kernel_1D(size=21, sigma=2.0):
    ax = torch.linspace(-(size // 2), size // 2, size)
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / torch.sum(kernel)  # Normalize the kernel
    return kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

def convolution_1D(signal, kernel, padding_mode ='zeros'):
    padding = kernel.shape[-1] // 2
    if padding_mode == 'zeros':
        return torch.nn.functional.conv1d(signal, kernel, padding=padding)
    elif padding_mode == 'circular':
        signal_padded = torch.cat([signal[:, :, -padding:], signal, signal[:, :, :padding]], dim=2)
        return torch.nn.functional.conv1d(signal_padded, kernel, padding=0)
    elif padding_mode == 'reflect':
        signal_padded = torch.nn.functional.pad(signal, (padding, padding), mode='reflect')
        return torch.nn.functional.conv1d(signal_padded, kernel, padding=0)
    elif padding_mode == 'replicate':
        signal_padded = torch.nn.functional.pad(signal, (padding, padding), mode='replicate')
        return torch.nn.functional.conv1d(signal_padded, kernel, padding=0)
    elif padding_mode == None:
        return torch.nn.functional.conv1d(signal, kernel)
    else:   
        raise ValueError(f"Unsupported padding mode: {padding_mode}")

def deconvolution_1D(signal, kernel):
    signal_ft = torch.fft.fft(signal)
    kernel_ft = torch.fft.fft(kernel, n=signal.shape[-1])
    kernel_ft = torch.where(torch.abs(kernel_ft) < 1e-10, torch.tensor(1e-10, device=kernel_ft.device), kernel_ft)
    deconvolved_ft = signal_ft / kernel_ft
    deconvolved = torch.fft.ifft(deconvolved_ft).real
    return deconvolved.reshape(-1, 1)

def degrade_image_1D(signal_, kernel = gaussian_normalised_kernel_1D(), noise_scale=1000, padding_mode='zeros'):
    if torch.any(signal_ > 1):
        raise ValueError("Input signal values should be in the range [0, 1]")
    signal = signal_.unsqueeze(0).unsqueeze(0)
    blurred = convolution_1D(signal, kernel, padding_mode=padding_mode)
    noisy_blurred = add_poisson_noise(blurred, scale_factor=noise_scale)
    return noisy_blurred.squeeze()