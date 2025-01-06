import numpy as np
from tqdm import tqdm

from goalvis.bbox import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """
    Class responsible for assigning the ball to the nearest player in each frame.
    """

    def __init__(self) -> None:
        """
        Initialize the PlayerBallAssigner with a maximum distance threshold.
        """
        self.max_player_ball_distance = 70

    def assign_ball_to_player(
        self,
        players: dict[int, dict],
        ball_bbox: list[float]
    ) -> int:
        """
        Assign the ball to the closest player if within the threshold.

        :param players: Dictionary mapping player_id -> detection info (including bbox).
        :param ball_bbox: Bounding box for the ball [x1, y1, x2, y2].
        :return: The ID of the player to whom the ball is assigned, or -1 if none.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player


    def assign_ball_possession(
        self,
        tracks: dict,
    ) -> np.ndarray:
        """
        Assign ball possession to players and keep track of which team
        has possession in each frame.

        :param tracks: Dictionary containing player/ball/referee tracks per frame.
        :return: A numpy array where each element indicates which team has ball control in that frame.
        """
        team_ball_control = []
        for frame_num, player_track in enumerate(tqdm(tracks["players"], desc="Assigning ball possession")):
            ball_bbox = tracks["ball"][frame_num][1]["bbox"]
            assigned_player = self.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
            else:
                # In case no one has the ball, assume same team as previous frame
                if len(team_ball_control) > 0:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(-1)

        return np.array(team_ball_control, dtype=int)