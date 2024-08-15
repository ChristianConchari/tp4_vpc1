import cv2 as cv 
import numpy as np
import random
import psutil
import os
import time
from typing import Tuple
import cv2 as cv
from typing import List
from typing import List
import cv2 as cv
import psutil
import os
import time

def get_video_properties(capture: cv.VideoCapture) -> Tuple[float, int]:
    """
    Retrieves the properties of a video capture.

    Parameters:
        capture (cv.VideoCapture): The video capture object.

    Returns:
        Tuple[float, int]: A tuple containing the frames per second (fps) and the total number of frames in the video.

    """
    fps = capture.get(cv.CAP_PROP_FPS)
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    return fps, total_frames

def select_random_frames(capture: cv.VideoCapture, total_frames: int, N: int) -> np.ndarray:
    """
    Selects N random frames from a video capture.

    Parameters:
        capture (cv.VideoCapture): The video capture object.
        total_frames (int): The total number of frames in the video.
        N (int): The number of frames to select.

    Returns:
        np.ndarray: An array containing the selected frames.
    """
    # Randomly select N frames from the video
    selected_frames_indices = sorted(random.sample(range(total_frames), N))
    selected_frames = []

    for idx in selected_frames_indices:
        # Set the frame position to the selected index
        capture.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        if ret:
            selected_frames.append(frame)

    return np.stack(selected_frames)

def process_frame_naive(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Applies naive background subtraction to a frame.

    Parameters:
        frame (np.ndarray): The input frame.
        background (np.ndarray): The background frame.

    Returns:
        np.ndarray: The foreground mask obtained by subtracting the background from the frame.
    """
    diff = cv.absdiff(frame, background)
    _, fg_mask = cv.threshold(cv.cvtColor(diff, cv.COLOR_BGR2GRAY), 50, 255, cv.THRESH_BINARY)
    return fg_mask

def display_frames(frame: np.ndarray, fg_mask: np.ndarray, method_name: str) -> None:
    """
    Display the given frame and foreground mask using the specified method name.

    Parameters:
        frame (np.ndarray): The input frame to be displayed.
        fg_mask (np.ndarray): The foreground mask to be displayed.
        method_name (str): The name of the method used for background subtraction.

    Returns:
        None
    """
    cv.imshow(f'{method_name} - Frame', frame)
    cv.imshow(f'{method_name} - FG Mask', fg_mask)

def measure_performance(process: psutil.Process, start_total_time: float, frame_count: int, memory_usage: List[int]) -> None:
    """
    Measures the performance of a process by calculating average memory usage and overall frames per second (FPS).

    Parameters:
        process (psutil.Process): The process to measure performance for.
        start_total_time (float): The start time of the process in seconds.
        frame_count (int): The total number of frames processed.
        memory_usage (List[int]): A list of memory usage values for each frame.

    Returns:
        None
    """
    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    average_memory = np.mean(memory_usage) / (1024 * 1024)
    overall_fps = frame_count / total_time
    print(f"Average memory usage: {average_memory:.2f} MB")
    print(f"Overall FPS: {overall_fps:.2f} frames/second")

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
        fg_mask = process_frame_naive(frame, median_background)
        display_frames(frame, fg_mask, 'Naive Median')

        end_time = time.time()
        processing_time = end_time - start_time
        wait_time = max(int((frame_duration - processing_time) * 1000), 1)
        if cv.waitKey(wait_time) in [ord('q'), 27]:
            break

        frame_count += 1
        memory_usage.append(process.memory_info().rss)

    measure_performance(process, start_total_time, frame_count, memory_usage)
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

    measure_performance(process, start_total_time, frame_count, memory_usage)
    capture.release()
    cv.destroyAllWindows()
