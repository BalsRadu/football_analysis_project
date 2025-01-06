from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


class TeamAssigner:
    """
    Class that uses KMeans clustering on color patches to determine team colors.
    """

    def __init__(self) -> None:
        """
        Initialize the TeamAssigner. The `team_colors` will store the centroid of each team color,
        and `player_team_dict` maps player_id to an assigned team (int).
        """
        self.team_colors: dict[int, np.ndarray] = {}
        self.player_team_dict: dict[int, int] = {}
        self.kmeans = None

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """
        Reshape the image to 2D and fit a KMeans model with 2 clusters.

        :param image: The image patch (often top half of player's bbox).
        :return: Fitted KMeans model.
        """
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """
        Extract the top half of the player's bounding box and perform KMeans
        to identify the cluster for the player's jersey color.

        :param frame: The video frame from which the player's patch is extracted.
        :param bbox: Bounding box [x1, y1, x2, y2].
        :return: The (B, G, R) color centroid of the player's cluster.
        """
        # Crop the player's bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0 : int(image.shape[0] / 2), :]

        # Get cluster model
        kmeans = self.get_clustering_model(top_half_image)

        # Get labels
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Identify which cluster is the player's
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Player color is the centroid of the player's cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame: np.ndarray, player_detections: dict) -> None:
        """
        Assign approximate team colors using KMeans for the first frame's set of players.

        :param frame: The first frame of the video.
        :param player_detections: dictionary of player detections in that frame.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Fit KMeans for 2 distinct team clusters
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        # Store team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(
        self,
        frame: np.ndarray,
        player_bbox: list[float],
        player_id: int
    ) -> int:
        """
        Determine which team (1 or 2) the player belongs to.

        :param frame: The current video frame.
        :param player_bbox: Bounding box for the player [x1, y1, x2, y2].
        :param player_id: Unique ID of the player track.
        :return: The team number (1 or 2).
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Compute the player's color
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id
        
    def assign_goalkeeper_color(self, frames: list[np.ndarray], tracks: dict) -> None:
        """
        Assign unique colors to goalkeepers (GK) while filtering out invalid detections.
    
        This function processes goalkeeper detections across all frames and assigns 
        unique colors to each goalkeeper using a clustering approach. The following 
        steps are executed:
    
        1) Gather the color from each GK detection across all frames.
        2) Filter out GK detections whose color is very close to team colors 
           (treating them as players instead).
        3) If more than 2 unique GK IDs remain, demote extras to players 
           (since soccer typically has only 2 goalkeepers).
        4) Cluster the remaining GK detections (up to 2 IDs) into 2 color groups 
           via KMeans clustering.
        5) Assign a temporary cluster label to each remaining GK detection.
        6) Perform a majority vote per GK ID to consistently assign one cluster color.
    
        :param frames: List of video frames corresponding to each detection.
        :param tracks: Dictionary containing "goalkeepers" and "players" data. 
                       Each frame's "goalkeepers" data maps GK IDs to their bounding 
                       boxes and detection information.
        :return: None. The function modifies the `tracks` dictionary in place.
        """
        def color_distance(c1, c2):
            return np.linalg.norm(c1 - c2)
    
        threshold = 40.0  # Distance threshold to discard GK as "player"
        leftover_gk_colors = []
        leftover_gk_refs = []  # Each element: (frame_num, gk_id)
        to_remove = []  # GK detections to be converted into players
    
        steps = [
            "Gather GK detections and filter close colors",
            "Demote extra GKs to players (if needed)",
            "Cluster GK colors",
            "Assign GK colors consistently",
        ]
    
        with tqdm(total=len(steps), desc="Assigning GK Colors", unit="step") as pbar:
            # 1) Gather all GK detections and filter out those near team colors
            for frame_num, gk_dict in enumerate(tracks["goalkeepers"]):
                frame = frames[frame_num]
                for gk_id, info in gk_dict.items():
                    bbox = info["bbox"]
                    gk_color = self.get_player_color(frame, bbox)
    
                    # Distances to the two known team colors
                    dist1 = color_distance(gk_color, self.team_colors[1])
                    dist2 = color_distance(gk_color, self.team_colors[2])
    
                    # If the color is too close to a team color, treat as a normal player
                    if dist1 < threshold or dist2 < threshold:
                        to_remove.append((frame_num, gk_id))
                    else:
                        leftover_gk_colors.append(gk_color)
                        leftover_gk_refs.append((frame_num, gk_id))
    
            # Convert "false GK" detections into players
            for frame_num, gk_id in to_remove:
                if gk_id in tracks["goalkeepers"][frame_num]:
                    bbox = tracks["goalkeepers"][frame_num][gk_id]["bbox"]
                    del tracks["goalkeepers"][frame_num][gk_id]
    
                    # Decide which team is closer
                    gk_color = self.get_player_color(frames[frame_num], bbox)
                    d1 = color_distance(gk_color, self.team_colors[1])
                    d2 = color_distance(gk_color, self.team_colors[2])
                    team_id = 1 if d1 <= d2 else 2
    
                    tracks["players"][frame_num][gk_id] = {
                        "bbox": bbox,
                        "team": team_id,
                        "team_color": self.team_colors[team_id],
                    }
    
            pbar.update(1)  # Update progress bar for step 1
    
            # 2) Demote extra GKs if more than 2 remain
            unique_gk_ids = set(gk_id for _, gk_id in leftover_gk_refs)
            if len(unique_gk_ids) > 2:
                frames_count = defaultdict(int)
                for _, gk_id in leftover_gk_refs:
                    frames_count[gk_id] += 1
    
                top_two = sorted(unique_gk_ids, key=lambda gid: frames_count[gid], reverse=True)[:2]
                to_remove_extra = [gid for gid in unique_gk_ids if gid not in top_two]
    
                new_leftover_colors = []
                new_leftover_refs = []
                for color_vec, (frame_num, gk_id) in zip(leftover_gk_colors, leftover_gk_refs):
                    if gk_id in to_remove_extra:
                        if gk_id in tracks["goalkeepers"][frame_num]:
                            bbox = tracks["goalkeepers"][frame_num][gk_id]["bbox"]
                            del tracks["goalkeepers"][frame_num][gk_id]
    
                            d1 = color_distance(color_vec, self.team_colors[1])
                            d2 = color_distance(color_vec, self.team_colors[2])
                            team_id = 1 if d1 <= d2 else 2
                            tracks["players"][frame_num][gk_id] = {
                                "bbox": bbox,
                                "team": team_id,
                                "team_color": self.team_colors[team_id],
                            }
                    else:
                        new_leftover_colors.append(color_vec)
                        new_leftover_refs.append((frame_num, gk_id))
    
                leftover_gk_colors = new_leftover_colors
                leftover_gk_refs = new_leftover_refs
    
            if len(leftover_gk_colors) < 2:
                pbar.update(len(steps) - pbar.n)  # Complete remaining progress
                return
    
            pbar.update(1)  # Update progress bar for step 2
    
            # 3) Cluster GK colors
            leftover_gk_colors_arr = np.array(leftover_gk_colors)
            gk_kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
            gk_kmeans.fit(leftover_gk_colors_arr)
    
            for i, (frame_num, gk_id) in enumerate(leftover_gk_refs):
                cluster_label = gk_kmeans.predict([leftover_gk_colors[i]])[0]
                tracks["goalkeepers"][frame_num][gk_id]["temp_cluster"] = cluster_label
    
            pbar.update(1)  # Update progress bar for step 3
    
            # 4) Assign final GK colors
            cluster_counts_per_id = defaultdict(lambda: [0, 0])
            for frame_num, gk_dict in enumerate(tracks["goalkeepers"]):
                for gk_id, info in gk_dict.items():
                    if "temp_cluster" in info:
                        c = info["temp_cluster"]
                        cluster_counts_per_id[gk_id][c] += 1
    
            final_cluster_for_id = {}
            for gk_id, counts in cluster_counts_per_id.items():
                final_cluster_for_id[gk_id] = 0 if counts[0] >= counts[1] else 1
    
            for frame_num, gk_dict in enumerate(tracks["goalkeepers"]):
                for gk_id, info in gk_dict.items():
                    if gk_id in final_cluster_for_id:
                        chosen_cluster = final_cluster_for_id[gk_id]
                        info["keeper_color"] = gk_kmeans.cluster_centers_[chosen_cluster].tolist()
                        info.pop("temp_cluster", None)
    
            pbar.update(1)  # Update progress bar for step 4


    def assign_player_teams(
        self,
        frames: list[np.ndarray],
        tracks: dict,
    ) -> None:
        """
        Assign each player a team and team color.

        :param frames: List of video frames.
        :param tracks: dictionary containing player/ball/referee tracks per frame.
        :param team_assigner: A TeamAssigner instance.
        """
        # We assume we have at least one frame and at least one player detection
        self.assign_team_color(frames[0], tracks["players"][0])
        self.assign_goalkeeper_color(frames, tracks)

        # Now iterate through each frame and each player to assign a team
        for frame_num, player_track in enumerate(tqdm(tracks["players"], desc="Assigning teams")):
            for player_id, track in player_track.items():
                team = self.get_player_team(frames[frame_num], track["bbox"], player_id)
                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = self.team_colors[team]
