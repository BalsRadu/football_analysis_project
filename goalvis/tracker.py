import os
import pickle
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

from goalvis.bbox import get_bbox_width, get_center_of_bbox, get_foot_position


class Tracker:
    """
    Class that loads a YOLO model and performs detection and tracking
    of players, ball, and referees.
    """

    def __init__(self, model_path: str) -> None:
        """
        :param model_path: Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add_position_to_tracks(self, tracks: dict) -> None:
        """
        Add a "position" field to each detection in `tracks`, which is either
        the bounding box center (for the ball) or foot position (for players/referees).

        :param tracks: Dictionary of object tracks.
        """
        for obj_name, object_tracks in tqdm(tracks.items(), desc="Adding positions to tracks"):
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if obj_name == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[obj_name][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions: list[Dict[int, Any]]) -> list[Dict[int, Any]]:
        """
        Interpolate missing ball positions across frames.

        :param ball_positions: List of dicts, each containing the ball bounding box in a frame.
        :return: Updated list of dicts with missing ball bounding boxes interpolated.
        """
        extracted_positions = [frame_dict.get(1, {}).get("bbox", []) for frame_dict in ball_positions]
        df_ball_positions = pd.DataFrame(extracted_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        interpolated_positions = []
        for row in tqdm(df_ball_positions.to_numpy().tolist(), desc="Interpolating ball positions"):
            interpolated_positions.append({1: {"bbox": row}})

        return interpolated_positions

    def detect_frames(self, frames: list[np.ndarray]) -> list:
        """
        Perform object detection on batches of frames using YOLO.

        :param frames: List of frames (np.ndarrays).
        :return: A list of YOLO detection outputs for each frame.
        """
        batch_size = 20
        detections = []
        for i in tqdm(range(0, len(frames), batch_size), desc="Processing frames for detection"):
            batch = frames[i:i + batch_size]
            detections_batch = self.model.predict(batch, conf=0.1, device=self.device, verbose=False)
            detections.extend(detections_batch)
        return detections

    def get_object_tracks(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | None = None
    ) -> dict:
        """
        Get object tracks (players, ball, referees) for each frame.
        Optionally read/write from a stub file.

        :param frames: List of video frames.
        :param read_from_stub: Whether to read from a pickle stub file.
        :param stub_path: Path to the stub file.
        :return: A dictionary with keys "players", "ball", and "referees",
            each containing a list of dicts with tracked objects per frame.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"[INFO] Loading tracks from {stub_path}")  # Print message here
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        # Initialize tracks structure
        tracks = {
            "players": [],
            "ball": [],
            "referees": [],
            "goalkeepers": []
        }

        # Convert YOLO detection outputs to ByteTrack-based tracks
        cls_names = None
        cls_names_inv = None

        for frame_num, detection in enumerate(tqdm(detections, desc="Tracking objects")):
            if cls_names is None:
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

            # For each frame, initialize an empty dict
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})
            tracks["goalkeepers"].append({})

            # Convert to supervision detection format
            detections_supervision = sv.Detections.from_ultralytics(detection)

            # # Convert goalkeeper to "player" class
            # for object_idx, class_id in enumerate(detections_supervision.class_id):
            #     if cls_names[class_id] == "goalkeeper":
            #         detections_supervision.class_id[object_idx] = cls_names_inv["player"]

            # Track objects
            detections_with_tracks = self.tracker.update_with_detections(detections_supervision)

            # Parse tracked objects for players/referees
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # For players
                if cls_names[cls_id] == "player":
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                # For referees
                elif cls_names[cls_id] == "referee":
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                # For goalkeepers
                elif cls_names[cls_id] == "goalkeeper":
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

            # For the ball, we look directly in the original detections
            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    # Always store the ball under key 1
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, int, int],
        track_id: int | None = None
    ) -> np.ndarray:
        """
        Draw an ellipse at the bottom of the bbox, optionally with
        a rectangle overlay for the track_id.

        :param frame: Current frame to draw on.
        :param bbox: Bounding box [x1, y1, x2, y2].
        :param color: (B, G, R) color for drawing.
        :param track_id: Optional track_id to display on rectangle.
        :return: The modified frame.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(
        self,
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draw a triangle (pointing upwards) at the top of the bbox.

        :param frame: Current frame to draw on.
        :param bbox: Bounding box [x1, y1, x2, y2].
        :param color: (B, G, R) color for drawing.
        :return: The modified frame.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(
        self,
        frame: np.ndarray,
        frame_num: int,
        team_ball_control: np.ndarray
    ) -> np.ndarray:
        """
        Draw a semi-transparent rectangle on the top-right corner of the frame
        and print the ball control percentage for both teams.

        :param frame: Current frame to draw on.
        :param frame_num: The index of the current frame.
        :param team_ball_control: Numpy array containing which team controlled the ball per frame.
        :return: The modified frame.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Avoid division by zero
        if (team_1_frames + team_2_frames) == 0:
            return frame

        team_1 = team_1_frames / (team_1_frames + team_2_frames)
        team_2 = team_2_frames / (team_1_frames + team_2_frames)

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1 * 100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2 * 100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3
        )

        return frame

    def draw_annotations(
        self,
        video_frames: list[np.ndarray],
        tracks: dict,
        team_ball_control: np.ndarray
    ) -> list[np.ndarray]:
        """
        Draw ellipses/triangles for players, referees, ball, and also
        team ball control info on each frame.

        :param video_frames: List of original video frames.
        :param tracks: Dictionary of object tracks.
        :param team_ball_control: Array of teams that have ball control in each frame.
        :return: List of frames with all annotations drawn.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(tqdm(video_frames, desc="Drawing annotations")):
            frame_copy = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame_copy = self.draw_ellipse(frame_copy, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame_copy = self.draw_triangle(frame_copy, player["bbox"], (0, 0, 255))

            # Draw referees
            for _, referee in referee_dict.items():
                frame_copy = self.draw_ellipse(frame_copy, referee["bbox"], (0, 255, 255))

            # Draw goalkeepers
            for track_id, keeper in goalkeeper_dict.items():
                color = keeper.get("keeper_color", (255, 255, 255))
                frame_copy = self.draw_ellipse(frame_copy, keeper["bbox"], color, track_id)

            # Draw ball
            for _, ball in ball_dict.items():
                frame_copy = self.draw_triangle(frame_copy, ball["bbox"], (0, 255, 0))

            # Draw team ball control
            frame_copy = self.draw_team_ball_control(frame_copy, frame_num, team_ball_control)

            output_video_frames.append(frame_copy)

        return output_video_frames
