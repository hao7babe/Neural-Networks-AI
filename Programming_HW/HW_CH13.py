import numpy as np

def convolve(image, kernel):
    """
    Perform convolution on a 2D image with a 2D kernel.
    
    Args:
        image (numpy.ndarray): Input image (2D array).
        kernel (numpy.ndarray): Convolution filter (2D array).

    Returns:
        numpy.ndarray: Convolution result (2D array).
    """
    # Dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialize the output
    output = np.zeros((output_height, output_width))

    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest
            region = image[i:i + kernel_height, j:j + kernel_width]
            # Element-wise multiplication and summation
            output[i, j] = np.sum(region * kernel)

    return output


# Example input image and kernel
image = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Perform convolution
result = convolve(image, kernel)

# Print the result
print("Convolution Result:")
print(result)
