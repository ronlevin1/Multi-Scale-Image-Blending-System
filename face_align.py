import cv2
import mediapipe as mp
import numpy as np

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
    idx_left_eye  = 33
    idx_right_eye = 263
    idx_nose      = 1
    return np.stack([
        landmarks[idx_left_eye],
        landmarks[idx_right_eye],
        landmarks[idx_nose],
    ], axis=0)

def align_face(src_img, dst_img):
    src_landmarks = get_landmarks(src_img)
    dst_landmarks = get_landmarks(dst_img)

    src_pts = get_affine_keypoints(src_landmarks)
    dst_pts = get_affine_keypoints(dst_landmarks)

    M = cv2.getAffineTransform(src_pts, dst_pts)
    h, w = dst_img.shape[:2]
    aligned = cv2.warpAffine(src_img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return aligned

if __name__ == "__main__":
    imgA = cv2.imread("images/eyal.jpg")  # base image, where mask is defined
    imgB = cv2.imread("images/bazz.jpg")  # image to align to A

    imgB_aligned = align_face(imgB, imgA)
    cv2.imwrite("images/bazz_aligned.jpg", imgB_aligned)

    # now use imgA, imgB_aligned and your single mask in `ex3.py`