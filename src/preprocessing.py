import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# bg_subtractor = cv2.createBackgroundSubtractorMOG2()


def convert_to_grayscale(frame: np.ndarray, use_builtin: bool = False) -> np.ndarray:
    """
    Convert an RGB frame to grayscale.
    Parameters:
    - frame: np.ndarray, input frame in RGB format.
    - use_builtin: bool, whether to use OpenCV's built-in function for conversion.
    Returns:
    - np.ndarray, frame in grayscale format.
    """
    
    if use_builtin:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # BGR weights (reversed from RGB weights)
    weights = np.array([0.1140, 0.5870, 0.2989])
    grayscale = np.dot(frame[..., :3], weights)
    return np.uint8(grayscale)


def apply_kernel(frame: np.ndarray, kernel: np.ndarray, use_builtin: bool = True, shadow_threshold: int = 250) -> np.ndarray:
    """
    Apply a 2D convolution kernel to a frame from scratch.
    
    Parameters:
    - frame: np.ndarray, input frame, already converted to grayscale.
    - kernel: np.ndarray, 2D kernel to apply.
    - use_builtin: bool, whether to use OpenCV's built-in filter2D
    - shadow_threshold: int, pixels below this value are considered shadows and clamped to 0
    
    Returns:
    - np.ndarray, frame with the kernel applied.
    """
    if use_builtin:
        filtered = cv2.filter2D(frame, -1, kernel)
        # Apply shadow thresholding
        filtered[filtered < shadow_threshold] = 0
        return filtered
    
    # Get dimensions
    frame_height, frame_width = frame.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding needed
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Initialize output array
    output = np.zeros_like(frame)
    
    # Iterate through each pixel in the image
    for i in range(pad_height, frame_height - pad_height):
        for j in range(pad_width, frame_width - pad_width):
            # Extract the region of interest
            roi = frame[i - pad_height:i + pad_height + 1,
                       j - pad_width:j + pad_width + 1]
            # Apply the kernel
            result = np.sum(roi * kernel)
            # Make sure to:
            # 1. Properly handle edge pixels (padding)
            # 2. Clamp values to [0, 255] range
            output[i, j] = np.clip(result, 0, 255)
            if output[i, j] < shadow_threshold:
                output[i, j] = 0
            output[i, j] = output[i, j].astype(np.uint8)
    
    return output


def process_video(input_path: str, filter_type: str, output_path: str = None, display: bool = True) -> None:
    """
    Process a video with a specified filter.
    
    Parameters:
    - input_path: str, path to input video file
    - filter_type: str, type of filter to apply ('blur', 'edge', 'sharpen')
    - output_path: str, path for output video (optional)
    - display: bool, whether to display the processed frames
    
    Returns:
    - None
    """
    # Define filter kernels
    kernels = {
        'blur': np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]]) / 16.0,
        'edge': np.array([[-1, -2, 0, 2, 1],
                         [-4, -8, 0, 8, 4],
                         [-6, -12, 0, 12, 6],
                         [-4, -8, 0, 8, 4],
                         [-1, -2, 0, 2, 1]]),
        'sharpen': np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
    }
    
    if filter_type not in kernels:
        raise ValueError(f"Invalid filter type. Choose from: {list(kernels.keys())}")
    
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter if output path is specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), False)
    
    # Initialize previous frame
    ret, prev_frame = cap.read()
    if not ret:
        return
    prev_gray = convert_to_grayscale(prev_frame)
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert current frame to grayscale
        gray_frame = convert_to_grayscale(frame)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray_frame, prev_gray)
        
        # Threshold the difference to get significant changes
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply the original filter
        filtered_frame = apply_kernel(gray_frame, kernels[filter_type])
        
        # Get coordinates of motion pixels
        motion_pixels = np.column_stack(np.where(motion_mask > 0))
        
        # Perform DBSCAN if we have enough motion pixels
        rectangles = []
        if len(motion_pixels) > 10:
            # Normalize the coordinates
            scaler = StandardScaler()
            motion_pixels_scaled = scaler.fit_transform(motion_pixels)
            
            # Apply DBSCAN
            db = DBSCAN(eps=0.3, min_samples=5).fit(motion_pixels_scaled)
            labels = db.labels_
            
            # Process each cluster
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                    
                # Get points in current cluster
                cluster_points = motion_pixels[labels == label]
                
                if len(cluster_points) > 10:  # Minimum cluster size
                    # Find bounding rectangle
                    min_y, min_x = np.min(cluster_points, axis=0)
                    max_y, max_x = np.max(cluster_points, axis=0)
                    rectangles.append((min_x, min_y, max_x, max_y))
        
        # Draw rectangles on the filtered frame
        display_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)
        for rect in rectangles:
            min_x, min_y, max_x, max_y = rect
            cv2.rectangle(display_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        
        # Write frame if output is specified
        if out:
            out.write(cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY))
        
        # Display the frames
        if display:
            scale_factor = 0.35
            display_width = int(gray_frame.shape[1] * scale_factor)
            display_height = int(gray_frame.shape[0] * scale_factor)
            
            # Resize frames
            gray_display = cv2.resize(gray_frame, (display_width, display_height))
            motion_display = cv2.resize(motion_mask, (display_width, display_height))
            result_display = cv2.resize(display_frame, (display_width, display_height))
            
            # Convert grayscale frames to BGR for consistent dimensions
            gray_display = cv2.cvtColor(gray_display, cv2.COLOR_GRAY2BGR)
            motion_display = cv2.cvtColor(motion_display, cv2.COLOR_GRAY2BGR)
            
            # Combine frames horizontally
            combined_frame = np.hstack((gray_display, motion_display, result_display))
            cv2.imshow('Original | Motion | Result', combined_frame)
            
            if current_frame % 50 == 0:
                print(f"Processed frame {current_frame} of {total_frames}")
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Update previous frame
        prev_gray = gray_frame
        current_frame += 1
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    if display:
        cv2.destroyAllWindows()
        

# Example usage:
if __name__ == "__main__":
    # Process image with blur filter
    process_video(
        input_path='./data/jump_one_slowed.mp4',
        filter_type='edge',
        output_path='./results/edge.mp4'
    )

