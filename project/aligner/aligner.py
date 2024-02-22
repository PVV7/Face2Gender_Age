import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
from numpy.linalg import inv, lstsq
from numpy.linalg import matrix_rank as rank
from project.utils import xywh2xyxy


class Aligner(object):
    def __init__(self):
        self._standart_facial_points = np.array(
                                        [[30.2946, 51.6963],
                                         [65.5318, 51.6963],
                                         [48.0252, 71.7366],
                                         [33.5493, 92.3655],
                                         [62.7299, 92.3655]], dtype=np.float32)

    def _find_non_reflective_similarity(self,
                                        uv: np.ndarray,
                                        xy: np.ndarray,
                                        K=2) -> np.ndarray:
        M = xy.shape[0]
        x = xy[:, 0].reshape((-1, 1))
        y = xy[:, 1].reshape((-1, 1))

        tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        X = np.vstack((tmp1, tmp2))

        u = uv[:, 0].reshape((-1, 1))
        v = uv[:, 1].reshape((-1, 1))
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
                   bbox: List[np.ndarray],
                   face_wh=(96, 112)
                   ) -> np.ndarray:

        x, y, r, b = map(round, bbox)
        x, y, r, b = xywh2xyxy((x, y, r, b))

        landmark_n = np.array(landmark, dtype=np.float32)

        crop_face_image = img[int(y):int(b), int(x):int(r)]
        crop_h, crop_w = crop_face_image.shape[:2]
        face_wh_n = np.array([face_wh], dtype=np.float32)

        rate_n = face_wh_n / np.array([crop_w, crop_h], dtype=np.float32)
        landmark_adj = (landmark_n - np.array([x, y], dtype=np.float32)) * rate_n
        crop_face_image = cv2.resize(crop_face_image, dsize=face_wh)
        trans_matrix = self._find_non_reflective_similarity(landmark_adj, self._standart_facial_points)

        max_standard_side = max(face_wh)
        aligned_face = cv2.warpAffine(crop_face_image.copy(), trans_matrix, (max_standard_side, max_standard_side))
        return aligned_face

    def _filter_kpts(self, kpts: np.ndarray) -> np.ndarray:

        kpts = np.resize(kpts, (5, 3))
        new_kpts = kpts[:, :2]

        return new_kpts

    def _face_info(self, box_and_points: List[Any]) -> List[Dict]:

        face_info = [{'landmark': self._filter_kpts(obj['kpts']),
                     'bbox': obj['boxes']}
                     for obj in box_and_points]

        return face_info

    def align_faces(self,
                    image: np.ndarray,
                    face_info: List[Dict]
                    ) -> List[np.ndarray]:
        assert len(face_info) > 0, 'There are no people in the image'

        faces = []
        box_and_points = face_info
        img = image

        face_infos = self._face_info(box_and_points)

        for face_info in face_infos:
            landmark = face_info["landmark"]
            bbox = face_info["bbox"]

            face = self._align_face(img, landmark, bbox)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            faces.append(face)

        return faces

    # def save_crop(self):
    #     faces = self.align_faces()
    #     if len(faces) == 0:
    #         return f'лиц нет'
    #
    #     for i, crop in enumerate(faces):
    #         crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    #         cv2.imwrite('croped_image_' + str(i) + '.jpg', crop)