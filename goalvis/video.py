import cv2
from tqdm import tqdm


def read_video(video_path: str) -> list:
    """
    Read a video from a given path and return a list of frames.

    :param video_path: Path to the input video file.
    :return: A list of frames (np.ndarrays).
    """
    print(f"[INFO] Reading video from {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames: list, output_video_path: str) -> None:
    """
    Save a list of frames as a video to the specified output path.

    :param output_video_frames: The list of frames to be saved.
    :param output_video_path: The desired path for the output video.
    """
    if not output_video_frames:
        raise ValueError("No frames to save.")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    for frame in tqdm(output_video_frames, desc="Saving video"):
        out.write(frame)
    out.release()
