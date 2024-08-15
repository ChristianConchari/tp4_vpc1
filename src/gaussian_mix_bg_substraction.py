import cv2 as cv
import numpy as np

import psutil
import os
import time

from typing import List
from .video_processing_functions import get_video_properties, display_frames

def mog2_or_knn_bg_subtraction(path: str, method: str = 'MOG2') -> None:
    """
    Apply background subtraction using either MOG2 or KNN method.
    
    Parameters:
        path (str): The path to the video file.
        method (str): The background subtraction method to use. Default is 'MOG2'.
    Raises:
        IOError: If there is an error opening the file.
    Returns:
        None
    """
    # Create the background subtractor object
    back_sub = cv.createBackgroundSubtractorMOG2() if method == 'MOG2' else cv.createBackgroundSubtractorKNN()

    # Open the video file
    capture = cv.VideoCapture(path)
    if not capture.isOpened():
        raise IOError(f"Error opening file: {path}")

    # Get the frames per second and the total number of frames
    fps_video, _ = get_video_properties(capture)
    
    # Calculate wait time in milliseconds
    wait_time = int(1000 / fps_video) 

    # Initialize variables
    frame_count = 0
    memory_usage = []
    total_processing_time = []
    
    # Start the total timer
    start_total_time = time.time()
    
    # Get the process object for memory usage
    process = psutil.Process(os.getpid())
    
    while True:
        # Read the next frame
        ret, frame = capture.read()
        if frame is None:
            break
        
        # Start the timer
        start_time = time.time()
        
        # Apply background subtraction
        fg_mask = back_sub.apply(frame)
        
        # Display the frames
        display_frames(frame, fg_mask, method)
        
        # End the timer
        end_time = time.time()
        
        # Calculate the processing time
        processing_time = end_time - start_time
        total_processing_time.append(processing_time)
        
        if cv.waitKey(wait_time) in [ord('q'), 27]:
            break
        
        # Increment the frame count
        frame_count += 1
        
        # Update the memory usage
        memory_usage.append(process.memory_info().rss)

    # End the total timer
    end_total_time = time.time()
    
    # Calculate the total time
    total_time = end_total_time - start_total_time
    
    # Calculate the average memory usage and overall FPS
    average_memory = np.mean(memory_usage) / (1024 * 1024)
    overall_fps = frame_count / total_time
    overall_processing_time = np.mean(total_processing_time)
    
    print(f"Average memory usage: {average_memory:.2f} MB")
    print(f"Overall FPS: {overall_fps:.2f} frames/second")
    print(f"Average processing time: {overall_processing_time:.6f} seconds")
    
    capture.release()
    cv.destroyAllWindows()
