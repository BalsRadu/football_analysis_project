from typing import Dict

import cv2
from tqdm import tqdm

from goalvis.bbox import get_foot_position, measure_distance


class SpeedAndDistanceEstimator:
    """
    Class to calculate speed and distance covered by objects (e.g. players).
    """

    def __init__(self) -> None:
        """
        Initialize the SpeedAndDistanceEstimator with frame window and frame rate.
        """
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks: dict) -> None:
        """
        Add the speed (in km/h) and distance covered (in meters) to each track in `tracks`.

        :param tracks: Dictionary of object tracks (players, ball, referees).
        """
        total_distance = {}

        for obj_name, object_tracks in tqdm(tracks.items(), desc="Calculating speed and distance"):
            # Skip ball and referees
            if obj_name in ("ball", "referees"):
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    # If no detection for this ID in the "last_frame", skip
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]["position_transformed"]
                    end_position = object_tracks[last_frame][track_id]["position_transformed"]

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    if obj_name not in total_distance:
                        total_distance[obj_name] = {}

                    if track_id not in total_distance[obj_name]:
                        total_distance[obj_name][track_id] = 0.0

                    total_distance[obj_name][track_id] += distance_covered

                    # Update speed/distance for all frames in the batch
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[obj_name][frame_num_batch]:
                            continue
                        tracks[obj_name][frame_num_batch][track_id]["speed"] = speed_km_per_hour
                        tracks[obj_name][frame_num_batch][track_id]["distance"] = total_distance[obj_name][track_id]

    def draw_speed_and_distance(
        self,
        frames: list,
        tracks: dict
    ) -> list:
        """
        Draw speed and distance next to the player bounding boxes in each frame.

        :param frames: The list of video frames.
        :param tracks: The dictionary of object tracks.
        :return: A list of frames with speed and distance annotations.
        """
        output_frames = []
        for frame_num, frame in enumerate(tqdm(frames, desc="Drawing speed and distance")):
            frame_copy = frame.copy()
            for obj_name, object_tracks in tracks.items():
                if obj_name in ("ball", "referees"):
                    continue

                for _, track_info in object_tracks[frame_num].items():
                    speed = track_info.get("speed", None)
                    distance = track_info.get("distance", None)
                    bbox = track_info.get("bbox", None)

                    if speed is None or distance is None or bbox is None:
                        continue

                    pos = get_foot_position(bbox)
                    pos_text = (pos[0], pos[1] + 40)

                    cv2.putText(
                        frame_copy,
                        f"{speed:.2f} km/h",
                        pos_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )
                    cv2.putText(
                        frame_copy,
                        f"{distance:.2f} m",
                        (pos_text[0], pos_text[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )

            output_frames.append(frame_copy)

        return output_frames
