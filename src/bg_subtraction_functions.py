import cv2 as cv
import numpy as np

import random
import psutil
import os
import time

from typing import Tuple, List
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


def naive_median_bg_subtraction(filename: str, N: int = 30) -> None:
    """
    Perform naive median background subtraction on a video file.
    
    Parameters:
        filename (str): The path to the video file.
        N (int, optional): The number of frames to randomly select for background calculation. Defaults to 30.
    Raises:
        IOError: If there is an error opening the video file.
    Returns:
        None
    """
    capture = cv.VideoCapture(filename)
    if not capture.isOpened():
        raise IOError(f"Error opening file: {filename}")

    fps_video, total_frames = get_video_properties(capture)
    selected_frames_array = select_random_frames(capture, total_frames, N)
    median_background = np.median(selected_frames_array, axis=0).astype(np.uint8)

    capture.set(cv.CAP_PROP_POS_FRAMES, 0)  # Go back to the start of the video

    frame_count = 0
    process = psutil.Process(os.getpid())
    memory_usage = []
    start_total_time = time.time()

    frame_duration = 1 / fps_video  # Expected duration of each frame in seconds

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        start_time = time.time()
        fg_mask = _process_frame_naive(frame, median_background)
        display_frames(frame, fg_mask, 'Naive Median')

        end_time = time.time()
        processing_time = end_time - start_time
        wait_time = max(int((frame_duration - processing_time) * 1000), 1)
        if cv.waitKey(wait_time) in [ord('q'), 27]:
            break

        frame_count += 1
        memory_usage.append(process.memory_info().rss)

    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    average_memory = np.mean(memory_usage) / (1024 * 1024)
    overall_fps = frame_count / total_time
    print(f"Average memory usage: {average_memory:.2f} MB")
    print(f"Overall FPS: {overall_fps:.2f} frames/second")
    
    capture.release()
    cv.destroyAllWindows()

def mog2_or_knn_bg_subtraction(filename: str, method: str = 'MOG2') -> None:
    """
    Apply background subtraction using either MOG2 or KNN method.
    
    Parameters:
        filename (str): The path to the video file.
        method (str): The background subtraction method to use. Default is 'MOG2'.
    Raises:
        IOError: If there is an error opening the file.
    Returns:
        None
    """
    back_sub = cv.createBackgroundSubtractorMOG2() if method == 'MOG2' else cv.createBackgroundSubtractorKNN()

    capture = cv.VideoCapture(filename)
    if not capture.isOpened():
        raise IOError(f"Error opening file: {filename}")

    fps_video, _ = get_video_properties(capture)
    wait_time = int(1000 / fps_video)  # Calculate wait time in milliseconds

    frame_count = 0
    total_time = 0
    process = psutil.Process(os.getpid())
    memory_usage: List[int] = []
    start_total_time = time.time()

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        start_time = time.time()
        fg_mask = back_sub.apply(frame)
        display_frames(frame, fg_mask, method)

        end_time = time.time()
        total_time += (end_time - start_time)
        if cv.waitKey(wait_time) in [ord('q'), 27]:
            break

        frame_count += 1
        memory_usage.append(process.memory_info().rss)

    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    average_memory = np.mean(memory_usage) / (1024 * 1024)
    overall_fps = frame_count / total_time
    print(f"Average memory usage: {average_memory:.2f} MB")
    print(f"Overall FPS: {overall_fps:.2f} frames/second")
    
    capture.release()
    cv.destroyAllWindows()
