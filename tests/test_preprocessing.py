import pytest
import numpy as np
import cv2
from src.preprocessing import convert_to_grayscale, apply_kernel


def test_grayscale_conversion_matches_builtin():
    # Create a sample RGB image
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Convert using both methods
    custom_gray = convert_to_grayscale(test_image, use_builtin=False)
    builtin_gray = convert_to_grayscale(test_image, use_builtin=True)

    # Compare results
    # Allow small differences due to floating point calculations
    np.testing.assert_allclose(custom_gray, builtin_gray, rtol=1e-2, atol=1)


def test_grayscale_shape():
    # Test that output shape is correct (height, width) without color channel
    test_image = np.random.randint(0, 256, (50, 75, 3), dtype=np.uint8)

    result = convert_to_grayscale(test_image)

    assert result.shape == (50, 75)
    assert len(result.shape) == 2


def test_grayscale_range():
    # Test that output values are in valid range [0, 255]
    test_image = np.random.randint(0, 256, (30, 40, 3), dtype=np.uint8)

    result = convert_to_grayscale(test_image)

    assert np.all(result >= 0)
    assert np.all(result <= 255)


def test_kernel_application_matches_builtin():
    # Create a sample grayscale image
    test_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    
    # Define some common kernels to test
    kernels = [
        # Gaussian blur kernel
        np.array([[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]], dtype=np.float32) / 16,
        # Sobel x-direction kernel
        np.array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]], dtype=np.float32)
    ]
    
    for kernel in kernels:
        # Apply kernel using custom implementation
        custom_result = apply_kernel(test_image, kernel)
        
        # Apply kernel using OpenCV's built-in function
        # Note: cv2.filter2D expects kernel to be float32
        builtin_result = cv2.filter2D(test_image, -1, kernel)
        
        # Compare results (allowing small differences due to floating point calculations)
        # Exclude border pixels as they might be handled differently
        border = 1
        custom_center = custom_result[border:-border, border:-border]
        builtin_center = builtin_result[border:-border, border:-border]
        
        np.testing.assert_allclose(custom_center, builtin_center, rtol=1e-2, atol=1)


def test_kernel_output_shape():
    # Test that output shape matches input shape
    test_image = np.random.randint(0, 256, (30, 40), dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.float32) / 9  # Simple averaging kernel
    
    result = apply_kernel(test_image, kernel)
    
    assert result.shape == test_image.shape
    assert len(result.shape) == 2


def test_kernel_range():
    # Test that output values are in valid range for uint8
    test_image = np.random.randint(0, 256, (20, 25), dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.float32) / 9  # Simple averaging kernel
    
    result = apply_kernel(test_image, kernel)
    
    assert np.all(result >= 0)
    assert np.all(result <= 255)
