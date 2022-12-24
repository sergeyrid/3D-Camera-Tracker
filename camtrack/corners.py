#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from copy import deepcopy
from tqdm import tqdm

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class Corner:
    def __init__(self, corner_id=0, corner_position=(0, 0), corner_quality=0, level=0):
        self.id = corner_id
        self.position = corner_position
        self.quality = corner_quality
        self.level = level

    def __gt__(self, other):
        return self.quality > other.quality


def _track_corners(image_0, image_1, corners_data, levels, quality_level=0.02):
    if len(corners_data) > 0:
        corner_coords = np.array(list(map(lambda corner: corner.position, corners_data))).reshape(-1, 2)
        min_threshold = float(
            min(
                map(
                    lambda x: max(
                        list(
                            map(
                                lambda corner: corner.quality,
                                filter(
                                    lambda corner: corner.level == x,
                                    corners_data))) + [0]),
                    range(levels)))) * quality_level
        corner_coords, states, _ = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255), np.uint8(image_1 * 255),
                                                            corner_coords, None,
                                                            winSize=(15, 15), maxLevel=levels,
                                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                      10, 0.03),
                                                            minEigThreshold=min_threshold)
    else:
        corner_coords = np.array([])
        states = np.array([])

    for i in range(len(corner_coords)):
        corners_data[i].position = corner_coords[i]
    corners_data = list(map(lambda pair: pair[0], filter(lambda pair: pair[1], zip(corners_data, states))))

    return deepcopy(corners_data)


def _find_corners(image_1, max_id, min_distance, mask, level=0, max_corners=500, quality_level=0.02):
    new_corner_coords, new_corner_qualities = cv2.goodFeaturesToTrackWithQuality(
        image_1, max_corners, quality_level, min_distance, np.uint8(mask))
    new_corners_data = []
    if new_corner_coords is not None:
        new_corner_coords = new_corner_coords.reshape(-1, 2)
        new_corners_data = [Corner(max_id + i + 1, new_corner_coords[i] * 2**level, new_corner_qualities[i], level)
                            for i in range(new_corner_coords.shape[0])]
    max_id += len(new_corners_data)
    _, max_quality = cv2.goodFeaturesToTrackWithQuality(image_1, 1, 0.1, 0, np.uint8(np.ones_like(image_1)))
    max_quality = max_quality[0]
    new_corners_data = list(filter(lambda corner: corner.quality >= max_quality * quality_level, new_corners_data))
    return new_corners_data, max_id


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    np.random.seed(42)
    pyr_levels = 2
    track_quality_level = 0.02
    find_quality_level = 0.02
    max_corners = 1000
    if 100 < len(frame_sequence) < 400:
        pyr_levels = 3
    corners_data = []
    image_0 = np.array([[]])
    max_id = 0
    circle_size = max(int(frame_sequence[0].shape[0] / 70), 5)
    for frame, image in tqdm(enumerate(frame_sequence)):
        image_1 = deepcopy(image)
        corners_data = _track_corners(image_0, image_1, corners_data, pyr_levels, track_quality_level)
        mask = np.ones_like(image).astype(np.uint8)
        for i in range(pyr_levels):
            min_distance = max(int(image_1.shape[0] / 50), 5)
            corner_coords = np.array(list(map(lambda corner: corner.position, corners_data)))
            for coord in corner_coords:
                mask = cv2.circle(mask, (int(coord[0] / 2**i), int(coord[1] / 2**i)),
                                  min_distance, 0.0, cv2.FILLED)
            new_corners_data, max_id = _find_corners(image_1, max_id, min_distance, mask,
                                                     i, max_corners, find_quality_level)
            corners_data += new_corners_data

            mask = cv2.pyrDown(mask)
            image_1 = cv2.pyrDown(image_1)
            if image_0.shape[1] != 0:
                image_0 = cv2.pyrDown(image_0)

        builder.set_corners_at_frame(frame, FrameCorners(
            np.array(list(map(lambda corner: corner.id, corners_data))),
            np.array(list(map(lambda corner: corner.position, corners_data))),
            np.array(list(map(lambda corner: 2**corner.level * circle_size, corners_data)))))
        image_0 = deepcopy(image)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
