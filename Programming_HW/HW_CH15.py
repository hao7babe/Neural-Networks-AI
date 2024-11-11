import numpy as np
from scipy.signal import convolve2d

def depthwise_convolution(image, kernel):
    """
    Perform depthwise convolution.
    Each channel of the image is convolved with a corresponding kernel.
    """
    if image.ndim != 3 or kernel.ndim != 3:
        raise ValueError("Image and kernel must have 3 dimensions (H, W, C).")
    
    # Ensure the number of channels in the image matches the kernel
    if image.shape[-1] != kernel.shape[-1]:
        raise ValueError("Number of channels in image and kernel must match.")
    
    # Apply convolution on each channel independently
    convoluted = np.zeros_like(image)
    for c in range(image.shape[-1]):
        convoluted[..., c] = convolve2d(image[..., c], kernel[..., c], mode='same', boundary='fill', fillvalue=0)
    return convoluted

def pointwise_convolution(image, kernel):
    """
    Perform pointwise convolution.
    A (1x1) kernel is applied across all channels of the image.
    """
    if kernel.ndim != 2 or kernel.shape[0] != 1 or kernel.shape[1] != image.shape[-1]:
        raise ValueError("Kernel must have shape (1, C_in) for pointwise convolution.")
    
    # Sum across channels with the pointwise kernel
    convoluted = np.sum(image * kernel[0, :, None, None].transpose(1, 2, 0), axis=-1)
    return convoluted

def perform_convolution(image, kernel, mode="depthwise"):
    """
    Perform either depthwise or pointwise convolution based on the mode flag.
    """
    if mode == "depthwise":
        return depthwise_convolution(image, kernel)
    elif mode == "pointwise":
        return pointwise_convolution(image, kernel)
    else:
        raise ValueError("Mode must be 'depthwise' or 'pointwise'.")

# Example usage
if __name__ == "__main__":
    # Example input image (3D array: H x W x C)
    image = np.random.rand(5, 5, 3)  # 5x5 image with 3 channels
    
    # Example kernel for depthwise convolution
    depthwise_kernel = np.random.rand(3, 3, 3)  # 3x3 kernel for each of 3 channels
    
    # Example kernel for pointwise convolution
    pointwise_kernel = np.random.rand(1, 3)  # (1, C_in) kernel for pointwise
    
    # Perform depthwise convolution
    depthwise_result = perform_convolution(image, depthwise_kernel, mode="depthwise")
    print("Depthwise Convolution Result:")
    print(depthwise_result)
    
    # Perform pointwise convolution
    pointwise_result = perform_convolution(image, pointwise_kernel, mode="pointwise")
    print("\nPointwise Convolution Result:")
    print(pointwise_result)
