import cv2 as cv 
import numpy as np
import random
from typing import Tuple

def get_video_properties(capture: cv.VideoCapture) -> Tuple[float, int]:
    """
    Retrieves the properties of a video capture.

    Parameters:
        capture (cv.VideoCapture): The video capture object.

    Returns:
        Tuple[float, int]: A tuple containing the frames per second (fps) and the total number of frames in the video.

    """
    # Get the frames per second and the total number of frames
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
