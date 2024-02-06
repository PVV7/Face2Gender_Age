import numpy as np
import cv2
from typing import List, Tuple, Dict
from numpy.linalg import inv, lstsq
from numpy.linalg import matrix_rank as rank
import matplotlib.pyplot as plt
#TODO: разобраться с typing во всех методах

class Aligner(object):
    def __init__(self, model):
        self.model = model
        self.result = self.model.detect()
        self.image = self.model.image
        self.standart_facial_points = np.array(
                                        [[30.2946, 51.6963],
                                         [65.5318, 51.6963],
                                         [48.0252, 71.7366],
                                         [33.5493, 92.3655],
                                         [62.7299, 92.3655]], dtype=np.float32)

    def _xywh2xyxy(self, box_xywh):
        cx, cy, w, h = box_xywh

        left_x = cx - w * 0.5
        top_y = cy - h * 0.5
        right_x = cx + w * 0.5
        bottom_y = cy + h * 0.5

        return left_x, top_y, right_x, bottom_y

    def _find_non_reflective_similarity(self, uv, xy, K=2):
        M = xy.shape[0]
        x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
        y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

        tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        X = np.vstack((tmp1, tmp2))

        u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
        v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
        U = np.vstack((u, v))

        # We know that X * r = U
        if rank(X) >= 2 * K:
            r, _, _, _ = lstsq(X, U, rcond=None)
            r = np.squeeze(r)
        else:
            raise Exception('cp2tform:twoUniquePointsReq')

        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]

        Tinv = np.array([
            [sc, -ss, 0],
            [ss, sc, 0],
            [tx, ty, 1]
        ])
        T = inv(Tinv)
        T[:, 2] = np.array([0, 0, 1])
        T = T[:, 0:2].T
        return T

    def _align_face(self,
                   img: np.ndarray,
                   landmark: List[Tuple],
                   bbox: List[float],
                   face_wh: Tuple = (96, 112)
                   ) -> np.ndarray:

        x, y, r, b = map(round, bbox)
        x, y, r, b = self._xywh2xyxy((x, y, r, b))

        landmark_n = np.array(landmark, dtype=np.float32)

        crop_face_image = img[int(y):int(b), int(x):int(r)]
        crop_h, crop_w = crop_face_image.shape[:2]
        face_wh_n = np.array([face_wh], dtype=np.float32)

        rate_n = face_wh_n / np.array([crop_w, crop_h], dtype=np.float32)
        landmark_adj = (landmark_n - np.array([x, y], dtype=np.float32)) * rate_n
        crop_face_image = cv2.resize(crop_face_image, dsize=face_wh)

        trans_matrix = self._find_non_reflective_similarity(landmark_adj, self.standart_facial_points)

        max_standard_side = max(face_wh)
        aligned_face = cv2.warpAffine(crop_face_image.copy(), trans_matrix, (max_standard_side, max_standard_side))
        return aligned_face

    def _filter_kpts(self, kpts):

        kpts = np.resize(kpts, (5, 3))
        new_kpts = kpts[:, :2]

        return new_kpts

    def _face_info(self, box_and_points):

        face_info = [{'landmark': self._filter_kpts(obj['kpts']),
                     'bbox': obj['boxes']}
                     for obj in box_and_points]


        return face_info

    def align_faces(self) -> List[np.ndarray]:
        faces = []

        box_and_points = self.result
        img = self.image

        face_infos = self._face_info(box_and_points)

        for face_info in face_infos:
            landmark = face_info["landmark"]
            bbox = face_info["bbox"]

            face = self._align_face(img, landmark, bbox)
            faces.append(face)

        return faces