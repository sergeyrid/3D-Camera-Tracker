#! /usr/bin/env python3

# __all__ = [
#     'track_and_calc_colors'
# ]

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
        return ids, points3d, np.array([])
    good_ids = ids[inliers]
    new_ids, new_points3d = zip(*list(filter(lambda p: p[0] in good_ids or p[0] not in ids,
                                             zip(old_ids, old_points3d))))
    return np.array(new_ids).astype(np.int64), np.array(new_points3d), \
        rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)


def triangulate_points(points2d, view_mats, intrinsic_mat):
    if points2d.shape[0] == 0:
        return np.array([])
    ans = []
    for i in range(points2d.shape[1]):
        coefs = []
        for point, view_mat in zip(points2d[:, i], view_mats):
            mat = intrinsic_mat @ view_mat
            coefs.append(mat[2] * point[0] - mat[0])
            coefs.append(mat[2] * point[1] - mat[1])
        coefs = np.array(coefs)
        x = np.linalg.lstsq(coefs[:, :3], -coefs[:, 3], rcond=None)[0]
        ans.append(x[:3])
    return np.array(ans)


def retriangulate(view_mats, corner_storage, intrinsic_mat, old_ids, old_points3d):
    print('Starting retriangulation')
    view_mats_with_ids = list(zip(range(len(view_mats)), view_mats))
    view_mats_with_ids = np.array(list(filter(lambda p: p[1].size != 0, view_mats_with_ids)), dtype=object)
    chosen_ids, chosen_view_mats = zip(*view_mats_with_ids[np.random.choice(
        list(range(len(view_mats_with_ids))), size=2, replace=False)])
    chosen_ids = list(chosen_ids)
    chosen_view_mats = list(chosen_view_mats)
    id1 = chosen_ids[0]
    id2 = chosen_ids[1]
    corners1 = corner_storage[id1]
    corners2 = corner_storage[id2]
    points2d = []
    correspondences = build_correspondences(corners1, corners2)

    points_ids = correspondences.ids
    points_num = len(points_ids)
    to_keep = []
    for i in range(points_num):
        to_keep.append(points_ids[i] in old_ids)
    points_ids = points_ids[to_keep]
    points_num = len(points_ids)
    points1 = correspondences.points_1[to_keep]
    points2 = correspondences.points_2[to_keep]
    points2d.append(points1)
    points2d.append(points2)
    for new_id, view_mat in view_mats_with_ids:
        if new_id in chosen_ids:
            continue
        new_corners = corner_storage[new_id]
        new_points_with_ids = list(filter(lambda p: p[0] in points_ids, zip(new_corners.ids, new_corners.points)))
        if len(new_points_with_ids) == points_num:
            new_points = np.array(list(zip(*new_points_with_ids))[1])
            chosen_ids.append(new_id)
            chosen_view_mats.append(view_mat)
            points2d.append(new_points)
    print(f'    Number of chosen frames: {len(chosen_ids)}')
    print(f'    Number of points: {len(points_ids)}')

    new_points3d = triangulate_points(np.array(points2d), chosen_view_mats, intrinsic_mat)
    result_points3d = np.copy(old_points3d)
    for i in range(len(points_ids)):
        if points_ids[i] in old_ids:
            result_points3d[np.where(old_ids == points_ids[i])[0]] = new_points3d[i]
    return result_points3d


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
    new_points3d_with_ids += list(zip(old_ids, old_points3d))
    ids_list, points_list = zip(*sorted(new_points3d_with_ids))
    return np.array(ids_list).astype(np.int64), np.array(points_list)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    print('Initialising')
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
    print(f'Initial frames: {frame1}, {frame2}')

    ids, points3d = triangulate(frame1, view_mats, corner_storage, intrinsic_mat,
                                triangulate_params, np.array([]), np.array([]).reshape(0, 3))
    print(f'Initial size of point cloud: {len(ids)}\n')

    print('Processing good frames')
    print('-----------------------------------')
    potential_ids = [frame1 - 1, frame1 + 1, frame2 - 1, frame2 + 1]
    while len(potential_ids) != 0:
        potential_ids = list(set(potential_ids))
        for i, frame_id in enumerate(potential_ids):
            if frame_id < 0 or frame_id >= frame_count or view_mats[frame_id].size != 0:
                potential_ids[i] = -1
        potential_ids = list(filter(lambda x: x != -1, potential_ids))
        if len(potential_ids) == 0:
            break
        print('Choosing between frames', end=' ')
        print(*potential_ids, sep=', ')
        intersec_cnts = []
        for frame_id in potential_ids:
            intersec_cnts.append(get_points(frame_id, corner_storage, ids, points3d)[0].shape[0])
        i = np.argmax(intersec_cnts)
        frame_id = potential_ids[i]
        if intersec_cnts[i] == 0:
            break
        print(f'Processing frame {frame_id}:')
        ids, points3d, view_mats[frame_id] = get_view(frame_id, corner_storage, intrinsic_mat, ids, points3d)
        if view_mats[frame_id].size == 0:
            print('    Failed to process')
            potential_ids.pop(i)
            continue
        ids, points3d = triangulate(frame_id, view_mats, corner_storage, intrinsic_mat,
                                    triangulate_params, ids, points3d)
        print(f'    New size of point cloud: {len(ids)}')
        print(f'    Processed frame {frame_id}')
        points3d = retriangulate(view_mats, corner_storage, intrinsic_mat, ids, points3d)
        potential_ids.pop(i)
        potential_ids.append(frame_id + 1)
        potential_ids.append(frame_id - 1)
        print('-----------------------------------')
    print('All good frames processed\n')

    print('Processing all frames')
    for i in range(len(view_mats)):
        print('Processing frame', i)
        _, _, view_mats[i] = get_view(i, corner_storage, intrinsic_mat, ids, points3d)
    print()

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
