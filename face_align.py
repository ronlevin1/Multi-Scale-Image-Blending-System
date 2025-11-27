import cv2
import mediapipe as mp
import numpy as np
from create_mask import FACE_OVAL_IDX

mp_face = mp.solutions.face_mesh


def get_landmarks(img):
    with mp_face.FaceMesh(static_image_mode=True,
                          max_num_faces=1,
                          refine_landmarks=True) as face_mesh:
        h, w = img.shape[:2]
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise RuntimeError("No face detected")
        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        return pts


def get_affine_keypoints(landmarks):
    # use: left eye outer corner, right eye outer corner, tip of nose
    idx_left_eye = 33
    idx_right_eye = 263
    idx_nose = 1
    return np.stack([
        landmarks[idx_left_eye],
        landmarks[idx_right_eye],
        landmarks[idx_nose],
    ], axis=0)


def get_shape_keypoints(landmarks, idxs=FACE_OVAL_IDX):
    return np.asarray(landmarks[idxs], dtype=np.float32)


def align_face(src_img, dst_img):
    """Align src_img to dst_img using facial landmarks."""
    src_landmarks = get_landmarks(src_img)
    dst_landmarks = get_landmarks(dst_img)

    src_shape = get_shape_keypoints(src_landmarks)
    dst_shape = get_shape_keypoints(dst_landmarks)

    M, _ = cv2.estimateAffinePartial2D(src_shape, dst_shape, method=cv2.LMEDS)

    if M is None:
        src_pts = get_affine_keypoints(src_landmarks)
        dst_pts = get_affine_keypoints(dst_landmarks)
        M = cv2.getAffineTransform(src_pts, dst_pts)

    h, w = dst_img.shape[:2]
    aligned = cv2.warpAffine(src_img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return aligned


def flip_lr(img):
    return cv2.flip(img, 1)


if __name__ == "__main__":
    imgB = cv2.imread("images/buzzi-vs-shauli/shauli.jpg")  # base image, where mask is defined
    imgA = flip_lr(cv2.imread("images/buzzi-vs-shauli/buzzi.jpeg"))  # image to align to A

    imgB_aligned = align_face(imgB, imgA)
    cv2.imwrite("images/buzzi-vs-shauli/shauli_aligned.jpg", imgB_aligned)
    #
    # imgC = flip_lr(cv2.imread("images/bazz_civil.jpg"))
    # imgC_aligned = align_face(imgC, imgA)
    # cv2.imwrite("images/bazz_civil_aligned.jpg", imgC_aligned)
