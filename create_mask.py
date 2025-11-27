import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

def build_face_mask(img, idxs=None, kernel_size=21):
    if idxs is None:
        idxs = FACE_OVAL_IDX

    h, w = img.shape[:2]
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                          refine_landmarks=True) as face_mesh:
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            raise RuntimeError("No face detected")
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idxs], dtype=np.int32)

    pts = cv2.convexHull(pts)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel)
    mask = (mask > 0).astype(np.uint8)

    return 1 - mask


if __name__ == "__main__":
    # Generate and save a face mask for the given image
    img = cv2.imread('images/buzzi-vs-shauli/buzzi.jpeg')
    mask = build_face_mask(img)

    # visualize alignment
    overlay = img.copy()
    overlay[mask == 0] = (0, 0, 0)  # or mask == 1 depending on which side you want

    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.imshow("Face overlay", blended)
    cv2.waitKey(0)          # wait for a key press
    cv2.destroyAllWindows() # close the window

    cv2.imwrite('images/buzzi-vs-shauli/buzzi_mask.jpg', mask * 255)