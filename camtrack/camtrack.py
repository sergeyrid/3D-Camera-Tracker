#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    build_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4
)


def get_points(i, corner_storage, ids, old_points3d):
    points2d_with_ids = zip(corner_storage[i].ids, corner_storage[i].points)
    points2d_with_ids = list(filter(lambda p: p[0] in ids, points2d_with_ids))
    new_ids = [p[0] for p in points2d_with_ids]
    points2d = np.array(list(map(lambda p: p[1], points2d_with_ids)))
    new_ids, points3d = zip(*list(filter(lambda p: p[0] in new_ids, zip(ids, old_points3d))))
    return np.array(new_ids).astype(np.int64), np.array(points3d), points2d


def get_view(i, corner_storage, intrinsic_mat, old_ids, old_points3d):
    ids, points3d, points2d = get_points(i, corner_storage, old_ids, old_points3d)
    if len(points3d) < 4:
        return np.array([])
    retval, r_vec, t_vec, inliers = cv2.solvePnPRansac(points3d, points2d, intrinsic_mat, None,
                                                       flags=cv2.SOLVEPNP_ITERATIVE)
    if not retval:
        print(':(')
        return ids, points3d, np.array([])
    good_ids = ids[inliers]
    new_ids, new_points3d = zip(*list(filter(lambda p: p[0] in good_ids or p[0] not in ids,
                                             zip(old_ids, old_points3d))))
    return np.array(new_ids).astype(np.int64), np.array(new_points3d), \
        rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)


def triangulate(frame1, view_mats, corner_storage, intrinsic_mat, triangulate_params, old_ids, old_points3d):
    if view_mats[frame1].size == 0:
        raise Exception('Invalid frame_id in triangulate')
    view_mats_with_ids = list(zip(range(len(view_mats)), view_mats))
    mat1 = view_mats_with_ids.pop(frame1)[1]
    view_mats_with_ids = list(filter(lambda p: p[1].size != 0, view_mats_with_ids))
    frame2, mat2 = view_mats_with_ids[np.random.choice(range(len(view_mats_with_ids)))]
    corrs = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    points3d, ids, med = triangulate_correspondences(corrs,
                                                     mat1,
                                                     mat2,
                                                     intrinsic_mat,
                                                     triangulate_params)
    new_points3d_with_ids = list(filter(lambda p: p[0] not in old_ids, zip(list(ids), list(points3d))))
    if len(new_points3d_with_ids) == 0:
        new_ids = np.array([])
        new_points3d = np.array([]).reshape(0, 3)
    else:
        new_ids, new_points3d = zip(*new_points3d_with_ids)
        new_ids = np.array(list(new_ids))
        new_points3d = np.array(list(new_points3d))
    ids_np, points_np = np.hstack((old_ids, new_ids)).astype(np.int64), np.concatenate((old_points3d, new_points3d),
                                                                                       axis=0)
    zipped = list(zip(list(ids_np), list(points_np)))
    ids_list, points_list = zip(*sorted(zipped))
    return np.array(ids_list).astype(np.int64), np.array(points_list)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    np.random.seed(42)
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [np.array([]) for _ in range(frame_count)]
    triangulate_params = TriangulationParameters(20, 1, 0.5)

    frame1 = known_view_1[0]
    frame2 = known_view_2[0]
    view_mats[frame1] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[frame2] = pose_to_view_mat3x4(known_view_2[1])

    ids, points3d = triangulate(frame1, view_mats, corner_storage, intrinsic_mat,
                                triangulate_params, np.array([]), np.array([]).reshape(0, 3))

    potential_ids = [frame1 - 1, frame1 + 1, frame2 - 1, frame2 + 1]
    print('FIRST ITERATION\n')
    while len(potential_ids) != 0:
        for i, frame_id in enumerate(potential_ids):
            if frame_id < 0 or frame_id >= frame_count or view_mats[frame_id].size != 0:
                potential_ids[i] = -1
        potential_ids = list(filter(lambda x: x != -1, potential_ids))
        if len(potential_ids) == 0:
            break
        intersec_cnts = []
        for frame_id in potential_ids:
            intersec_cnts.append(get_points(frame_id, corner_storage, ids, points3d)[0].shape[0])
        i = np.argmax(intersec_cnts)
        frame_id = potential_ids[i]
        if intersec_cnts[i] == 0:
            break
        print('Processing frame', frame_id)
        ids, points3d, view_mats[frame_id] = get_view(frame_id, corner_storage, intrinsic_mat, ids, points3d)
        if view_mats[frame_id].size == 0:
            potential_ids.pop(i)
            continue
        ids, points3d = triangulate(frame_id, view_mats, corner_storage, intrinsic_mat,
                                    triangulate_params, ids, points3d)
        potential_ids.pop(i)
        potential_ids.append(frame_id + 1)
        potential_ids.append(frame_id - 1)

    print('SECOND ITERATION')

    for i in range(len(view_mats)):
        print('Processing frame', i)
        _, _, view_mats[i] = get_view(i, corner_storage, intrinsic_mat, ids, points3d)

    point_cloud_builder = PointCloudBuilder(ids, points3d)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
