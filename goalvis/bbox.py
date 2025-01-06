import math


def get_center_of_bbox(bbox: list[float]) -> tuple[int, int]:
    """
    Compute the center point (x, y) of a given bounding box.

    :param bbox: A list [x1, y1, x2, y2].
    :return: The center point of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox: list[float]) -> float:
    """
    Compute the width of a given bounding box.

    :param bbox: A list [x1, y1, x2, y2].
    :return: The width of the bounding box.
    """
    return bbox[2] - bbox[0]


def measure_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """
    Compute the Euclidean distance between two points.

    :param p1: The (x, y) coordinates of the first point.
    :param p2: The (x, y) coordinates of the second point.
    :return: The Euclidean distance between p1 and p2.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def measure_xy_distance(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    """
    Compute the difference in x and y coordinates between two points.

    :param p1: The (x, y) coordinates of the first point.
    :param p2: The (x, y) coordinates of the second point.
    :return: The differences (x_diff, y_diff).
    """
    return (p1[0] - p2[0], p1[1] - p2[1])


def get_foot_position(bbox: list[float]) -> tuple[int, int]:
    """
    Compute the approximate foot position of a player from a bounding box.

    :param bbox: A list [x1, y1, x2, y2].
    :return: The foot position (x, y).
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
