import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from scipy.ndimage import convolve1d

GAUSSIAN_VECTOR = [1, 4, 6, 4, 1]
GAUSSIAN_KERNEL = np.array(GAUSSIAN_VECTOR, dtype=np.float64)
REDUCE_KERNEL = GAUSSIAN_KERNEL / GAUSSIAN_KERNEL.sum()
EXPAND_KERNEL = GAUSSIAN_KERNEL / (GAUSSIAN_KERNEL.sum() / 2.0)

"------------------------------------------------------------------------------"
"----------- Helper functions from face_align.py and create_mask.py -----------"
"------------------------------------------------------------------------------"

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
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idxs],
                       dtype=np.int32)

    pts = cv2.convexHull(pts)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel)
    mask = (mask > 0).astype(np.uint8)

    return 1 - mask


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


"------------------------------------------------------------------------------"
"-------------------------- Task 1 - Blended Image ----------------------------"
"------------------------------------------------------------------------------"


def _blur_single_channel(img, kernel):
    # TODO: check results with new implementation
    temp = convolve1d(img, kernel, axis=1, mode='nearest')
    blurred = convolve1d(temp, kernel, axis=0, mode='nearest')
    return blurred
    # OLD VERSION BELOW:
    # pad = len(kernel) // 2
    # temp = np.zeros_like(img, dtype=np.float64)
    #
    # row_padded = np.pad(img, ((0, 0), (pad, pad)), mode='edge')
    # for i in range(img.shape[0]):
    #     temp[i, :] = np.convolve(row_padded[i], kernel, mode='valid')
    #
    # col_padded = np.pad(temp, ((pad, pad), (0, 0)), mode='edge')
    # blurred = np.zeros_like(img, dtype=np.float64)
    # for j in range(img.shape[1]):
    #     blurred[:, j] = np.convolve(col_padded[:, j], kernel, mode='valid')
    #
    # return blurred


def blur(img, kernel):
    img = img.astype(np.float64, copy=False)
    if img.ndim == 2:
        return _blur_single_channel(img, kernel)
    blurred = np.zeros_like(img)
    for c in range(img.shape[2]):
        blurred[..., c] = _blur_single_channel(img[..., c], kernel)
    return blurred


def reduce(img):
    # blur and sub-sample
    smoothed = blur(img, REDUCE_KERNEL)
    return smoothed[::2, ::2]


def expand(img):
    expanded_shape = (img.shape[0] * 2, img.shape[1] * 2) + (
        () if img.ndim == 2 else (img.shape[2],))
    expanded_img = np.zeros(expanded_shape, dtype=np.float64)
    # pad image with zeros then blur
    expanded_img[::2, ::2, ...] = img
    return blur(expanded_img, EXPAND_KERNEL)


def gaussian_pyramid(img, num_of_levels):
    """Construct a Gaussian pyramid from the input image."""
    pyramid = [img]
    current_img = img

    for _ in range(1, num_of_levels):
        if min(current_img.shape[:2]) < 2:
            break
        reduced_img = reduce(current_img)
        pyramid.append(reduced_img)
        current_img = reduced_img

    return pyramid


def laplacian_pyramid(img, num_of_levels):
    """Construct a Laplacian pyramid from the input image."""
    gaussian_pyr = gaussian_pyramid(img, num_of_levels)
    laplacian_pyr = []

    for i in range(len(gaussian_pyr) - 1):
        expanded = expand(gaussian_pyr[i + 1])
        # Ensure the expanded image matches the size of the current Gaussian level
        if expanded.shape != gaussian_pyr[i].shape:
            target_height, target_width = gaussian_pyr[i].shape[:2]
            expanded = expanded[:target_height, :target_width, ...]
        laplacian_pyr.append(gaussian_pyr[i] - expanded)

    # last level is the same as in Gaussian pyramid
    laplacian_pyr.append(gaussian_pyr[-1])

    return laplacian_pyr


def load_image(img_path, as_gray=False):
    img = plt.imread(img_path).astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if as_gray and img.ndim == 3:
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    return img


def plot_triptych(imgA, imgB, blended, titles=None, figsize=(12, 4)):
    # decrease horizontal space between subplots
    if titles is None:
        titles = ("Buzzi", "Bibi (Aligned)", "Blended")
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # set horizontal spacing explicitly to avoid tight_layout warnings
    fig.subplots_adjust(wspace=0.00)
    for ax, image, title in zip(axes, (imgA, imgB, blended), titles):
        display = image.astype(np.float64, copy=False)
        if display.max() > 1.0:
            display = display / 255.0
        if display.ndim == 2:
            ax.imshow(display, cmap='gray')
        else:
            ax.imshow(np.clip(display, 0.0, 1.0))
        ax.set_title(title)
        ax.axis('off')
    return fig, axes


def max_pyramid_levels(shape):
    h, w = shape[:2]
    levels = 1
    while min(h, w) >= 2:
        h = (h + 1) // 2
        w = (w + 1) // 2
        levels += 1
    return levels


def pyramid_blending(imgA_path, imgB_path, output_path,
                     mask_path='inputs/binary_mask.png'):
    """
    • Given two inputs A and B, and a binary mask M
    • Construct Laplacian Pyramids La and Lb
    • Construct a Gaussian Pyramid from mask M - Gm
    • Create a third Laplacian Pyramid Lc where for each level k
        𝐿𝑐(𝑖, 𝑗) = 𝐺𝑚(𝑖, 𝑗)*𝐿𝑎(𝑖, 𝑗) + (1 − 𝐺𝑚(𝑖, 𝑗))*𝐿𝑏(𝑖, 𝑗)
    • Sum all levels Lc in to get the blended image
    """
    # load inputs and mask
    imgA = load_image(imgA_path)
    imgB = load_image(imgB_path)
    mask = load_image(mask_path, as_gray=True)
    mask = np.clip(mask, 0.0, 1.0)

    # equalize sizes
    min_shape = (min(imgA.shape[0], imgB.shape[0], mask.shape[0]),
                 min(imgA.shape[1], imgB.shape[1], mask.shape[1]))
    imgA = imgA[:min_shape[0], :min_shape[1], ...]
    imgB = imgB[:min_shape[0], :min_shape[1], ...]
    mask = mask[:min_shape[0], :min_shape[1]]

    # determine number of levels
    num_levels = min(max_pyramid_levels(imgA.shape),
                     max_pyramid_levels(imgB.shape),
                     max_pyramid_levels(mask.shape))

    # create pyramids for inputs & mask
    La = laplacian_pyramid(imgA, num_levels)
    Lb = laplacian_pyramid(imgB, num_levels)
    Gm = gaussian_pyramid(mask, num_levels)

    # create blended Laplacian pyramid Lc
    Lc = []
    for k in range(num_levels):
        Lc_level = Gm[k][..., None] * La[k] + (1 - Gm[k])[..., None] * Lb[k]
        Lc.append(Lc_level)

    # reconstruct blended image from Lc
    blended_img = Lc[-1]

    # collect intermediate reconstructions for plotting
    intermediate_imgs = [np.clip(blended_img.copy(), 0.0, 1.0)]

    for k in range(num_levels - 2, -1, -1):
        blended_img = expand(blended_img)
        # Ensure expanded image matches the size of current Lc level
        if blended_img.shape != Lc[k].shape:
            target_height, target_width = Lc[k].shape[:2]
            blended_img = blended_img[:target_height, :target_width, ...]
        blended_img += Lc[k]

        # plot intermediate results as subplots in the same figure
        # -> collect snapshot after adding this level
        intermediate_imgs.append(np.clip(blended_img.copy(), 0.0, 1.0))

    blended_img = np.clip(blended_img, 0.0, 1.0)

    # plot all intermediate reconstructions in a single row
    try:
        n = len(intermediate_imgs)
        if n > 0:
            fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
            if n == 1:
                axes = [axes]
            for i, img_snap in enumerate(intermediate_imgs):
                ax = axes[i]
                disp = img_snap
                if disp.ndim == 2:
                    ax.imshow(disp, cmap='gray')
                else:
                    ax.imshow(np.clip(disp, 0.0, 1.0))
                ax.set_title(f'Step {i}')
                ax.axis('off')
            fig.subplots_adjust(wspace=0.02)
            plt.tight_layout()
            fig.savefig('outputs/report/blended_intermediate_steps.jpg', bbox_inches='tight')
            plt.close(fig)
    except Exception:
        # keep pipeline robust; plotting errors shouldn't stop the pipeline
        pass

    plt.imsave(output_path, blended_img)
    return blended_img


# TODO: delete this function before submission
def plot_pyramid_levels(Lc, blended_img, num_levels):
    """
    for the report - section 4:
    create gaussian pyramid for blended image and with Lc (laplacian pyr)
    save as figures showing 8 levels in the pyramid, in 2 rows of 4 levels,
    positioned side-by-side. choose 8 levels from the middle of the pyramid.
    """

    def _select_middle_levels(pyr, count=8):
        total = len(pyr)
        if total >= count:
            start = max(0, (total - count) // 2)
            return pyr[start:start + count]
        # pad by repeating last level if not enough levels
        selected = list(pyr)
        while len(selected) < count:
            selected.append(pyr[-1])
        return selected

    def _to_display(img):
        img = np.asarray(img, dtype=np.float64)
        mn, mx = img.min(), img.max()
        # If the image is already in normalised [0,1] range just clip and return it.
        # This preserves correct appearance for Gaussian pyramid levels (G), avoiding
        # unnecessary contrast stretching that makes G[0] look almost white.
        if mn >= 0.0 and mx <= 1.0:
            return np.clip(img, 0.0, 1.0)

        # If dynamic range is effectively zero, center and amplify tiny variations.
        if mx - mn < 1e-12:
            centered = img - img.mean()
            s = np.abs(centered).max()
            if s < 1e-12:
                return np.zeros_like(img)  # truly constant
            centered /= (2 * s)
            return np.clip(centered + 0.5, 0.0, 1.0)

        # Otherwise scale to [0,1] (useful for Laplacian levels with negatives).
        return np.clip((img - mn) / (mx - mn), 0.0, 1.0)

    G_blended = gaussian_pyramid(blended_img, num_levels)
    # pick levels
    # G_levels = _select_middle_levels(G_blended, 8)
    # Lc_levels = _select_middle_levels(Lc, 8)
    G_levels = G_blended[3:7]
    Lc_levels = Lc[3:7]

    # save Gaussian pyramid of blended image
    plt.figure(figsize=(12, 6))
    for i, lvl in enumerate(G_levels):
        ax = plt.subplot(1, 4, i + 1)
        disp = lvl
        if disp.ndim == 2:
            ax.imshow(disp, cmap='gray')
        else:
            ax.imshow(disp)
        ax.set_title(f'G level {i + 1}')
        ax.axis('off')

    plt.subplots_adjust(wspace=0.06, hspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('inputs/report/blended_gaussian_pyr.jpg')
    plt.close()

    # save Laplacian pyramid (Lc) levels
    plt.figure(figsize=(12, 6))
    for i, lvl in enumerate(Lc_levels):
        ax = plt.subplot(1, 4, i + 1)
        disp = _to_display(lvl)
        if disp.ndim == 2:
            ax.imshow(disp, cmap='gray')
        else:
            ax.imshow(np.clip(disp, 0.0, 1.0))
        ax.set_title(f'L level {i + 1}')
        ax.axis('off')
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.06, hspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('outputs/report/blended_laplacian_pyr.jpg')
    plt.close()


def plot_fft_magnitude(image, out_path):
    """Compute log-magnitude of 2D FFT of image luminance and save as uint8 image."""
    img = np.array(image, dtype=np.float64, copy=False)
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = img if img.ndim == 2 else img[..., 0]
    # ensure numeric range suitable for FFT
    if gray.max() > 1.0:
        gray = gray / 255.0
    gray = np.clip(gray, 0.0, 1.0)

    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    magnitude = np.log1p(np.abs(Fshift))

    # normalize to [0,1]
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
    else:
        magnitude = np.zeros_like(magnitude)

    out_uint8 = (magnitude * 255.0).round().astype(np.uint8)

    # render with matplotlib to add a title (use vmin/vmax to avoid autoscaling)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(out_uint8, cmap='gray', vmin=0, vmax=255)
    ax.set_title('Buzzi vs. Bibi FFT Magnitude (log scale)')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


"------------------------------------------------------------------------------"
"--------------------------- Task 2 - Hybrid Image ----------------------------"
"------------------------------------------------------------------------------"


def gaussian_kernel(sigma_high, sigma_low):
    """
    Create Gaussian kernels for high-pass and low-pass filtering.
    """
    ksize = int(6 * sigma_low) | 1
    x = np.arange(ksize) - ksize // 2
    g_low = np.exp(-(x ** 2) / (2 * sigma_low ** 2))
    g_low /= g_low.sum()

    ksize = int(6 * sigma_high) | 1
    x = np.arange(ksize) - ksize // 2
    g_high = np.exp(-(x ** 2) / (2 * sigma_high ** 2))
    g_high /= g_high.sum()
    return g_high, g_low


def hybrid_image(imgA_path, imgB_path, output_path, gray_scale=False):
    """
    Create a hybrid image from imgA and imgB.
    """
    # constants
    LOW_SIGMA_RATIO = 0.02
    HIGH_SIGMA_RATIO = 0.005

    # load inputs
    if gray_scale:
        A = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE) / 255.0
        B = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE) / 255.0
    else:
        A = load_image(imgA_path)
        B = load_image(imgB_path)

    # match inputs size
    hA, wA = A.shape[:2]
    hB, wB = B.shape[:2]
    target_h = min(hA, hB)
    target_w = min(wA, wB)
    A = cv2.resize(A, (target_w, target_h), interpolation=cv2.INTER_AREA)
    B = cv2.resize(B, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # calculate sigmas & kernels
    base = min(target_h, target_w)
    sigma_low = LOW_SIGMA_RATIO * base
    sigma_high = HIGH_SIGMA_RATIO * base
    g_high, g_low = gaussian_kernel(sigma_high, sigma_low)

    # create hybrid image
    low_B = blur(B, g_low)
    blurred_A = blur(A, g_high)
    high_A = A - blurred_A

    hybrid = low_B + high_A
    hybrid = np.clip(hybrid, 0, 1)
    # plot in the same figure low_b and high_a
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(low_B, cmap='gray' if gray_scale else None)
    axes[0].set_title('Low-pass B', pad=10)
    axes[0].axis('off')

    axes[1].imshow(high_A + 0.5, cmap='gray' if gray_scale else None)
    axes[1].set_title('High-pass A (offset by 0.5 for visibility)', pad=10)
    axes[1].axis('off')

    axes[2].imshow(hybrid, cmap='gray' if gray_scale else None)
    axes[2].set_title('Hybrid Image', pad=10)
    axes[2].axis('off')

    # reserve top margin so titles are not clipped, tighten layout and save
    plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(wspace=0.02)
    plt.savefig('outputs/report/hybrid_low_high_components.jpg',
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)

    # display and save
    plt.figure(figsize=(8, 8))
    plt.imshow(hybrid, cmap='gray' if gray_scale else None)
    plt.axis('off')
    plt.show()
    plt.imsave(output_path, hybrid, cmap='gray' if gray_scale else None)

    return hybrid


"-----------------------------------------------------------------------------"
"------------------------------ Main Execution -------------------------------"
"-----------------------------------------------------------------------------"

if __name__ == '__main__':
    # GOOD BLENDING PATHS
    imgA_path = 'inputs/buzzi-vs-bibi/buzzi.jpeg'
    imgB_path = 'inputs/buzzi-vs-bibi/bibi.jpg'
    imgB_aligned_path = 'inputs/buzzi-vs-bibi/aligned.jpg'
    mask_path = 'inputs/buzzi-vs-bibi/mask.jpg'

    # BAD BLENDING PATHS
    # imgA_path = 'inputs/eyal-vs-buzz/eyal.jpg'
    # imgB_path = 'inputs/eyal-vs-buzz/bazz.jpg'
    # imgB_aligned_path = 'inputs/eyal-vs-buzz/aligned.jpg'
    # mask_path = 'inputs/eyal-vs-buzz/mask.jpg'

    # HYBRID IMAGE PATHS
    # imgA_path = 'inputs/buzzi-vs-shauli/buzzi.jpeg'
    # imgB_path = 'inputs/buzzi-vs-shauli/shauli.jpg'
    # imgB_aligned_path = 'inputs/buzzi-vs-shauli/aligned.jpg'
    # mask_path = 'inputs/buzzi-vs-shauli/mask.jpg'

    print("\nRunning...\n")

    imgA_bgr = cv2.imread(imgA_path)
    imgB_bgr = cv2.imread(imgB_path)
    if imgA_bgr is None or imgB_bgr is None:
        raise FileNotFoundError(
            'Could not load source inputs for blending pipeline')

    # -------------------------- Task 1 Execution ----------------------------

    # create mask
    mask = build_face_mask(imgA_bgr)
    cv2.imwrite(mask_path, mask * 255)

    # align imgB to imgA
    imgB_aligned = align_face(imgB_bgr, imgA_bgr)
    cv2.imwrite(imgB_aligned_path, imgB_aligned)

    # execute blending
    print("\nBlending inputs...\n")
    blended = pyramid_blending(imgA_path, imgB_aligned_path,
                               'inputs/eyal-vs-buzz/blended_ver3.jpg',
                               mask_path)
    plot_triptych(load_image(imgA_path), load_image(imgB_aligned_path),
                  blended)
    # plot_fft_magnitude(blended, 'inputs/report/blended_fft_magnitude.jpg')

    # save figure
    plt.savefig('outputs/results/trio_ver1.jpg')
    plt.show()

    # -------------------------- Task 2 Execution ----------------------------

    # print("Creating hybrid image...")
    # output_path = 'outputs/results/hybrid_ver5.jpg'
    # hybrid_image(imgA_path, imgB_aligned_path, output_path, gray_scale=True)

    print("\nDone ..!")

    # TODO: create a tar file you can run the following command:
    # tar -cvf ex3.tar ex3.py requirements.txt ./inputs ./outputs
