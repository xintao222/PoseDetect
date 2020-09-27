import numpy as np
import cv2


class AlignPoints():
    def __init__(self):
        self.align_size = [512, 512]
        self.projection_zoom = 2

    def get_points_edge(self, points):
        points = points[:, :, [0, 1]]
        points = points
        points_x = points[:, :, 0]
        points_y = points[:, :, 1]
        minx = np.min(points_x, axis=1)
        maxx = np.max(points_x, axis=1)
        miny = np.min(points_y, axis=1)
        maxy = np.max(points_y, axis=1)

        return points, minx, miny, maxx, maxy

    def get_transfrom_mat(self, x1, y1, x2, y2):
        zoom = self.projection_zoom
        s = self.align_size

        min_coord = np.stack([x1, y1], axis=1)
        max_coord = np.stack([x2, y2], axis=1)
        body_crop_center2d_origin = 0.5 * (min_coord + max_coord)
        fit_size = np.amax(
            np.maximum(max_coord - body_crop_center2d_origin, body_crop_center2d_origin - min_coord), axis=1)
        crop_size_best = 2 * fit_size * zoom
        bscale2d_origin = float(s[0]) / crop_size_best
        tmp1 = np.stack([bscale2d_origin, np.zeros([len(bscale2d_origin)]),
                         s[1] / 2 - bscale2d_origin * body_crop_center2d_origin[:, 0]], axis=1)
        tmp2 = np.stack([np.zeros([len(bscale2d_origin)]), bscale2d_origin,
                         s[0] / 2 - bscale2d_origin * body_crop_center2d_origin[:, 1]], axis=1)
        bH = np.stack([tmp1, tmp2], axis=1).astype(np.float32)
        return bH

    def transform_points(self, points, mat):
        tmp_points = points[:, :, :2]

        a = mat[:, :, :2]
        b = mat[:, :, 2:]
        b = np.repeat(b, axis=-1, repeats=points.shape[1]).transpose(0, 2, 1)
        ta = np.matmul(tmp_points, a, )
        points = ta + b
        return points

    def align_points(self, points):
        points, xmin, ymin, xmax, ymax = self.get_points_edge(points)

        transform_mat = self.get_transfrom_mat(xmin, ymin, xmax, ymax)
        transformed_points = self.transform_points(points, transform_mat)
        return transformed_points

    def transform_image(self, img, mat):
        transformed_image = cv2.warpAffine(
            img, mat, (self.align_size[1], self.align_size[0]), flags=cv2.INTER_LANCZOS4)

        return transformed_image
