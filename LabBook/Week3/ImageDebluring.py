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

def convolution(image, kernel, padding_mode='zeros'):    
    # Apply convolution with padding to maintain image size
    padding = (kernel.size(2) // 2, kernel.size(3) // 2)
    if padding_mode == 'zeros':
        filtered = F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
    elif padding_mode == 'circular':
        image_padded = torch.cat([image[:, -padding[0]:], image, image[:, :padding[0]]], dim=1)
        image_padded = torch.cat([image_padded[-padding[1]:, :], image_padded, image_padded[:padding[1], :]], dim=0)
        filtered = F.conv2d(image_padded.unsqueeze(0).unsqueeze(0), kernel, padding=0)
    elif padding_mode == 'reflect':
        image_padded = F.pad(image.unsqueeze(0).unsqueeze(0), (padding[1], padding[1], padding[0], padding[0]), mode='reflect')
        filtered = F.conv2d(image_padded, kernel, padding=0)
    elif padding_mode == 'replicate':
        image_padded = F.pad(image.unsqueeze(0).unsqueeze(0), (padding[1], padding[1], padding[0], padding[0]), mode='replicate')
        filtered = F.conv2d(image_padded, kernel, padding=0)
    elif padding_mode == None:
        filtered = F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel)
    else:   
        raise ValueError(f"Unsupported padding mode: {padding_mode}")
    
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
    ax = torch.arange(-size//2 + 1, size//2 + 1)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / torch.sum(kernel)
    print(f"kernel shape: {kernel.shape}")
    return kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, size, size]

def richardson_lucy(degraded_img, kernel, num_iters, estimate = None, c = 1e-10, padding_mode='zeros'):    
    if estimate is None:
        estimate = torch.ones_like(degraded_img)  # Initial estimate
    else:
        estimate = estimate.clone()  # Ensure the input estimate is not modified
    kernel_mirror = torch.flip(kernel, [-2, -1])  # Mirror the kernel

    # handles 1D and 2D kernels
    if len(kernel.shape) == 3:
        kernel_mirror = torch.flip(kernel, [-1])  
        conv_func = convolution_1D
    elif len(kernel.shape) == 4:
        kernel_mirror = torch.flip(kernel, [-2, -1])  
        conv_func = convolution
    else:
        raise ValueError("Unsupported kernel or image dimensions")
    
    for _ in range(num_iters):
        relative_blur = degraded_img / (conv_func(estimate, kernel, padding_mode=padding_mode) + c)  # c added to avoid division by zero
        estimate *= conv_func(relative_blur, kernel_mirror, padding_mode=padding_mode)

    return estimate

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
    return deconvolved