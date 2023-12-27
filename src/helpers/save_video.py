import cv2
import datetime
import os


def convert_numpy_to_video(agent_name, frame_list, output_path, fps=30):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{agent_name}_video_{timestamp}.mp4"
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
        for i, frame in enumerate(frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR color space if necessary

            # Overlay frame number on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner = (10, 30)
            font_scale = 0.5
            font_color = (0, 255, 0)  # Green color
            thickness = 2
            cv2.putText(frame, f"Frame: {i + 1}", bottom_left_corner, font, font_scale, font_color, thickness)

            out.write(frame)
    except Exception as e:
        # Handle any exceptions and release the VideoWriter
        out.release()
        raise e

    # Release the VideoWriter and close the video file
    out.release()

    print(f"Video saved in: {filepath}")
