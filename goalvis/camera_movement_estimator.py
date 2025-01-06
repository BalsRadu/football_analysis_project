import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from goalvis.bbox import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    """
    Class to estimate the camera movement from frame to frame using optical flow.
    """

    def __init__(self, frame: np.ndarray) -> None:
        """
        :param frame: The first frame of the video (in BGR format).
        """
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        # Example: Only look for features in some region
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(
        self,
        tracks: dict,
        camera_movement_per_frame: list[list[float]]
    ) -> None:
        """
        Add adjusted position to all tracks based on the computed camera movement.

        :param tracks: Dictionary of object tracks.
        :param camera_movement_per_frame: The list of (x, y) camera movements per frame.
        """
        for obj_name, object_tracks in tqdm(tracks.items(), desc="Adjusting positions"):
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1]
                    )
                    tracks[obj_name][frame_num][track_id]["position_adjusted"] = position_adjusted

    def get_camera_movement(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | None = None
    ) -> list[list[float]]:
        """
        Compute camera movement (x, y) for each frame using optical flow. If `read_from_stub`
        is True and `stub_path` is valid, the stub is loaded instead of running computation.

        :param frames: list of frames from the video.
        :param read_from_stub: Whether to read from a stub file.
        :param stub_path: Path to the stub pickle file.
        :return: A list of [x_movement, y_movement] for each frame.
        """
        # Read from stub if requested
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"[INFO] Loading camera movement from {stub_path}")  # Print message
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Add tqdm here to track progress for camera movement estimation
        for frame_num in tqdm(range(1, len(frames)), desc="Estimating camera movement"):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray,
                frame_gray,
                old_features,
                None,
                **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point,
                        new_features_point
                    )

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        # Write to stub if path provided
        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(
        self,
        frames: list[np.ndarray],
        camera_movement_per_frame: list[list[float]]
    ) -> list[np.ndarray]:
        """
        Draw camera movement (x, y) text on each frame.

        :param frames: list of video frames.
        :param camera_movement_per_frame: The list of (x, y) camera movements per frame.
        :return: list of frames with camera movement text drawn.
        """
        output_frames = []

        for frame_num, frame in enumerate(tqdm(frames, desc="Drawing camera movement")):
            frame_copy = frame.copy()
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), cv2.FILLED)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            cv2.putText(
                frame_copy,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )
            cv2.putText(
                frame_copy,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )

            output_frames.append(frame_copy)

        return output_frames
