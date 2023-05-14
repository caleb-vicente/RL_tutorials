import os

import cv2
import numpy as np
import datetime

def convert_numpy_to_video(frame_list, output_path, fps=30):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"video_{timestamp}.mp4"
    filepath = output_path + filename

    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Check if the frame_list is empty
    if len(frame_list) == 0:
        raise ValueError("Empty frame_list provided.")

    # Get frame dimensions from the first frame
    frame_height, frame_width, _ = frame_list[0].shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))

    try:
        # Iterate through each frame and write it to the video file
        for frame in frame_list:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR color space if necessary
            out.write(frame)
    except Exception as e:
        # Handle any exceptions and release the VideoWriter
        out.release()
        raise e

    # Release the VideoWriter and close the video file
    out.release()

    print(f"video saved in: {filepath}")