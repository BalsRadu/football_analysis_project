from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


class ViewTransformer:
    """
    A class that handles perspective transformation of points from
    the original video frame (pixel space) to a defined target space 
    (e.g., a bird's-eye or court view).

    :ivar pixel_vertices: An array of four (x, y) points defining the quadrilateral 
        in the image space that will be transformed.
    :ivar target_vertices: An array of four (x, y) points defining the target 
        quadrilateral in the transformed (court) space.
    :ivar perspective_transformer: The 3x3 matrix used by OpenCV to perform perspective
        transformation from pixel space to court space.
    """

    def __init__(self) -> None:
        """
        Initialize the ViewTransformer with default pixel/target vertices.

        The default vertices are hard-coded to map a known region (e.g., 
        the field or court) from the image space to a reference field space.
        """
        # Dimensions of the field/court, for example:
        court_width = 68.0
        court_length = 23.32

        # Four corner points in the source (pixel) space
        self.pixel_vertices: np.ndarray = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ], dtype=np.float32)

        # Corresponding four corner points in the target (transformed) space
        self.target_vertices: np.ndarray = np.array([
            [0, court_length],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ], dtype=np.float32)

        # Compute the perspective transform matrix
        self.perspective_transformer: np.ndarray = cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.target_vertices
        )

    def transform_point(self, point: np.ndarray) -> np.ndarray | None:
        """
        Transform a single point from pixel space to court space using
        the perspective transform matrix.

        :param point: A 2D point in (x, y) format.
        :return: The transformed point in the target space, or None if point is 
            outside of the defined region.
        """
        # Ensure we have integer (x, y)
        p = (int(point[0]), int(point[1]))

        # Check if point is inside the polygon defined by `pixel_vertices`
        # Negative values from pointPolygonTest mean the point is outside.
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # Reshape to match the input requirements of cv2.perspectiveTransform
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(
            reshaped_point, 
            self.perspective_transformer
        )

        # Flatten result from shape (1, 1, 2) -> (1, 2)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks: dict[str, Any]) -> None:
        """
        For each object's adjusted position in `tracks`, add a 
        `position_transformed` field corresponding to the transformed
        coordinate in the court space.

        :param tracks: A dictionary keyed by object type (e.g., "players",
            "ball", "referees"), each containing a list of per-frame 
            dictionaries of track info.
        """
        for obj_name, object_tracks in tqdm(tracks.items(), desc="Transforming positions"):
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    position_adjusted = track_info.get("position_adjusted", None)
                    if position_adjusted is None:
                        tracks[obj_name][frame_num][track_id]["position_transformed"] = None
                        continue

                    # Convert to ndarray for transform_point
                    position_array = np.array(position_adjusted, dtype=np.float32)
                    position_transformed = self.transform_point(position_array)

                    if position_transformed is not None:
                        # Squeeze to shape (2,) then convert to list
                        position_transformed = position_transformed.squeeze().tolist()

                    tracks[obj_name][frame_num][track_id]["position_transformed"] = position_transformed
