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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    max_corners = 250
    quality_level = 0.01
    min_distance = 15
    image_0 = frame_sequence[0]
    corner_coords = cv2.goodFeaturesToTrack(image_0, max_corners, quality_level, min_distance)
    corner_coords = corner_coords.reshape(-1, 2)
    corners = FrameCorners(
        np.arange(corner_coords.shape[0]),
        corner_coords,
        np.ones_like(corner_coords) * 30
    )
    builder.set_corners_at_frame(0, corners)
    max_corners = 25
    quality_level = 0.2
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        corner_coords, _, _ = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255), np.uint8(image_1 * 255),
                                                       corner_coords, None,
                                                       winSize=(15, 15), maxLevel=2,
                                                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                 10, 0.03))
        if frame % 5 == 0 and corner_coords is not None:
            mask = np.ones_like(image_1)
            for coord in corner_coords:
                mask = cv2.circle(mask, np.int8(coord), min_distance, 0, cv2.FILLED)
            new_corner_coords = cv2.goodFeaturesToTrack(image_1, max(0, max_corners - corner_coords.shape[0]),
                                                        quality_level, min_distance,
                                                        mask=np.uint8(mask))
            if new_corner_coords is not None:
                new_corner_coords = new_corner_coords.reshape(-1, 2)
            corner_coords = np.vstack((corner_coords, new_corner_coords))
        corners = FrameCorners(
            np.arange(corner_coords.shape[0]),
            corner_coords,
            np.ones_like(corner_coords) * 30
        )
        image_0 = image_1
        builder.set_corners_at_frame(frame, corners)


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
