import cv2 as cv
import numpy as np
from datetime import datetime
import psutil
import os
import time

from typing import List
from .video_processing_functions import get_video_properties, select_random_frames, display_frames

def _process_frame_naive(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Applies naive background subtraction to a frame.

    Parameters:
        frame (np.ndarray): The input frame.
        background (np.ndarray): The background frame.

    Returns:
        np.ndarray: The foreground mask obtained by subtracting the background from the frame.
    """
    # Calculate the absolute difference between the frame and the background
    diff = cv.absdiff(frame, background)
    
    # Convert the difference to a binary mask
    _, fg_mask = cv.threshold(cv.cvtColor(diff, cv.COLOR_BGR2GRAY), 50, 255, cv.THRESH_BINARY)
    
    return fg_mask


def naive_median_bg_subtraction(path: str, N: int = 30, interval: float = None) -> None:
    """
    Perform naive median background subtraction on a video file.
    
    Parameters:
        path (str): The path to the video file.
        N (int, optional): The number of frames to randomly select for background calculation. Defaults to 30.
        interval (float, optional): The time interval in seconds to recalculate the background. If None, no recalculation is performed. Defaults to None.
    
    Raises:
        IOError: If there is an error opening the video file.
    
    Returns:
        None
    """
    # Open the video file
    capture = cv.VideoCapture(path)
    if not capture.isOpened():
        raise IOError(f"Error opening file: {path}")

    # Start the timer
    start_time = time.time()
    
    # Get the frames per second and the total number of frames
    fps_video, total_frames = get_video_properties(capture)
    print(f"The video has {total_frames} frames and a frame rate of {fps_video} frames per second.")
    
    # Calculate the wait time in milliseconds
    wait_time = int(1000 / fps_video) 
    
    # Randomly select N frames from the video
    selected_frames_array = select_random_frames(capture, total_frames, N)

    # Calculate the median background frame
    median_background = np.median(selected_frames_array, axis=0).astype(np.uint8)
    
    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Initial background frame calculated in {elapsed_time:.2f} seconds")
    print("Starting background subtraction...")

    # Reset the video capture to the beginning of the video
    capture.set(cv.CAP_PROP_POS_FRAMES, 0)

    # Initialize variables
    frame_count = 0
    memory_usage = []
    total_processing_time = []
    
    # Start the total timer
    start_total_time = time.time()
    
    # Get the process object for memory usage
    process = psutil.Process(os.getpid())
    
    # Set the next background update time if interval is not None
    next_bg_update_time = start_total_time + interval if interval is not None else None
    
    while True:
        # Read the next frame
        ret, frame = capture.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # Calculate the elapsed video time
        elapsed_video_time = current_time - start_total_time
        elapsed_video_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_video_time))
        
        # Recalculate the background if the interval has passed and interval is not None
        if interval is not None and current_time >= next_bg_update_time:
            # Get the current frame position
            current_frame_pos = capture.get(cv.CAP_PROP_POS_FRAMES)

            print(f"Recalculating background frame at video time {elapsed_video_time_str}, after interval of {interval:.2f} seconds...")
            recalculating_start_time = time.time()
            
            selected_frames_array = select_random_frames(capture, total_frames, N)
            median_background = np.median(selected_frames_array, axis=0).astype(np.uint8)
            next_bg_update_time = current_time + interval
            
            recalculating_end_time = time.time()
            recalculating_time = recalculating_end_time - recalculating_start_time
            print(f"Background frame recalculated on {recalculating_time} seconds.")
            
            # Reset to the frame position before recalculation
            capture.set(cv.CAP_PROP_POS_FRAMES, current_frame_pos)
        
        # Start the timer
        start_time = time.time()
        
        # Apply naive median background subtraction if median_background is available
        if median_background is not None:
            fg_mask = _process_frame_naive(frame, median_background)
        else:
            fg_mask = np.zeros_like(frame)
        
        # Display the frames
        display_frames(frame, fg_mask, 'Naive Median')
        
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