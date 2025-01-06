import warnings

import cv2
import numpy as np

from goalvis.camera_movement_estimator import CameraMovementEstimator
from goalvis.player_ball_assigner import PlayerBallAssigner
from goalvis.speed_and_distance_estimator import SpeedAndDistanceEstimator
from goalvis.team_assigner import TeamAssigner
from goalvis.tracker import Tracker
from goalvis.video import read_video, save_video
from goalvis.view_transformer import ViewTransformer

warnings.filterwarnings("ignore")


def main() -> None:
    """
    Main entry point for the Goalvis pipeline. Reads a video,
    tracks objects, estimates camera movement, transforms positions,
    calculates speeds, assigns teams, determines ball possession,
    draws annotations, and finally saves an output video.
    """
    # Read video
    video_frames = read_video("data/samples/08fd33_4.mp4")

    # Initialize Tracker
    tracker = Tracker("models/yolo11l.pt")

    # Get object tracks
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="data/stubs/track_stubs.pkl"
    )

    # Add position (foot or center) to all tracks
    tracker.add_position_to_tracks(tracks)

    # Estimate camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path="data/stubs/camera_movement_stubs.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Transform positions to a "bird's-eye" view or similar
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Estimate speed and distance
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_player_teams(video_frames, tracks)

    # Assign ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = player_assigner.assign_ball_possession(tracks)

    # Draw output
    # 1) Draw object tracks on video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # 2) Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )

    # 3) Draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks
    )

    # Save final annotated video
    save_video(output_video_frames, "data/output_videos/output.avi")
    print("Output video saved successfully.")

if __name__ == "__main__":
    main()
