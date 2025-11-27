import numpy as np
import matplotlib.pyplot as plt
import cv2
from create_mask import build_face_mask
from face_align import align_face, flip_lr

GAUSSIAN_VECTOR = [1, 4, 6, 4, 1]
GAUSSIAN_KERNEL = np.array(GAUSSIAN_VECTOR, dtype=np.float64)
REDUCE_KERNEL = GAUSSIAN_KERNEL / GAUSSIAN_KERNEL.sum()
EXPAND_KERNEL = GAUSSIAN_KERNEL / (GAUSSIAN_KERNEL.sum() / 2.0)

"------------------------------------------------------------------------------"
"-------------------------- Task 1 - Blended Image ----------------------------"
"------------------------------------------------------------------------------"


def _blur_single_channel(img, kernel):
    pad = len(kernel) // 2
    temp = np.zeros_like(img, dtype=np.float64)

    row_padded = np.pad(img, ((0, 0), (pad, pad)), mode='edge')
    for i in range(img.shape[0]):
        temp[i, :] = np.convolve(row_padded[i], kernel, mode='valid')

    col_padded = np.pad(temp, ((pad, pad), (0, 0)), mode='edge')
    blurred = np.zeros_like(img, dtype=np.float64)
    for j in range(img.shape[1]):
        blurred[:, j] = np.convolve(col_padded[:, j], kernel, mode='valid')

    return blurred


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
    if titles is None:
        titles = ("Buzzi", "Shauli (Aligned)", "Blended")
    fig, axes = plt.subplots(1, 3, figsize=figsize)
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
    fig.tight_layout()
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
                     mask_path='images/binary_mask.png'):
    """
    • Given two images A and B, and a binary mask M
    • Construct Laplacian Pyramids La and Lb
    • Construct a Gaussian Pyramid from mask M - Gm
    • Create a third Laplacian Pyramid Lc where for each level k
        𝐿𝑐(𝑖, 𝑗) = 𝐺𝑚(𝑖, 𝑗)*𝐿𝑎(𝑖, 𝑗) + (1 − 𝐺𝑚(𝑖, 𝑗))*𝐿𝑏(𝑖, 𝑗)
    • Sum all levels Lc in to get the blended image
    """
    # load images and mask
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
    for k in range(num_levels - 2, -1, -1):
        blended_img = expand(blended_img)
        # Ensure the expanded image matches the size of the current Lc level
        if blended_img.shape != Lc[k].shape:
            target_height, target_width = Lc[k].shape[:2]
            blended_img = blended_img[:target_height, :target_width, ...]
        blended_img += Lc[k]

    blended_img = np.clip(blended_img, 0.0, 1.0)
    plt.imsave(output_path, blended_img)
    return blended_img


"------------------------------------------------------------------------------"
"--------------------------- Task 2 - Hybrid Image ----------------------------"
"------------------------------------------------------------------------------"


def gaussian_kernel(radius):
    """Return a 1D separable kernel based on discrete binomial coefficients."""
    if radius <= 0:
        return np.array([1], dtype=np.int64)
    size = int(2 * np.ceil(3 * radius) + 1)
    if size < 1:
        size = 1
    coeff = np.array([1], dtype=np.int64)
    for _ in range(size - 1):
        coeff = np.convolve(coeff, np.array([1, 1], dtype=np.int64))
    return coeff / coeff.sum()


def hybrid_image(imgA_path, imgB_path, output_path,
                 low_pass_radius, high_pass_radius):
    """
    Create a hybrid image by combining the low-frequency content of imgA
    with the high-frequency content of imgB.
    """
    imgA = load_image(imgA_path, as_gray=True)
    imgB = load_image(imgB_path, as_gray=True)

    # Ensure both images have 3 channels
    if imgA.ndim == 2:
        imgA = np.stack([imgA] * 3, axis=-1)
    if imgB.ndim == 2:
        imgB = np.stack([imgB] * 3, axis=-1)

    # Resize imgB to match imgA's size
    target_h, target_w = imgA.shape[:2]
    if imgB.shape[0] != target_h or imgB.shape[1] != target_w:
        imgB = cv2.resize(imgB, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    kernelA = gaussian_kernel(high_pass_radius)
    blurredA = blur(imgA, kernelA)
    high_pass_A = imgA - blurredA

    kernelB = gaussian_kernel(low_pass_radius)
    low_pass_B = blur(imgB, kernelB)

    hybrid_img = np.clip(low_pass_B + high_pass_A, 0.0, 1.0)
    # hybrid_img = low_pass_B + high_pass_A

    plt.figure(figsize=(8, 8))
    plt.imshow(hybrid_img)
    plt.axis('off')
    plt.show()
    plt.imsave(output_path, hybrid_img)

    return hybrid_img


"------------------------------------------------------------------------------"

if __name__ == '__main__':
    imgA_path = 'images/buzzi-vs-eyal/buzzi.jpeg'
    imgB_path = 'images/buzzi-vs-shauli/shauli.jpg'
    print("\nRunning...\n")

    """
    Task 1 Code - Blended Image
    """

    # imgB_aligned_path = 'images/buzzi-vs-shauli/aligned.jpg'
    # mask_path = 'images/buzzi-vs-shauli/mask.jpg'
    #
    # imgA_bgr = cv2.imread(imgA_path)
    # imgB_bgr = cv2.imread(imgB_path)
    # if imgA_bgr is None or imgB_bgr is None:
    #     raise FileNotFoundError(
    #         'Could not load source images for blending pipeline')
    #
    # mask = build_face_mask(imgA_bgr)
    # cv2.imwrite(mask_path, mask * 255)
    #
    # imgB_aligned = align_face(imgB_bgr, imgA_bgr)
    # cv2.imwrite(imgB_aligned_path, imgB_aligned)
    #
    # print("\nBlending images...\n")
    # blended = pyramid_blending(imgA_path, imgB_aligned_path,
    #                            'images/eyal-vs-buzz/blended_ver3.jpg',
    #                            mask_path)
    # plot_triptych(load_image(imgA_path), load_image(imgB_aligned_path),
    #               blended)
    # # save the figure
    # plt.savefig('images/outputs/trio_ver4.jpg')
    # plt.show()

    """ 
    Task 2 Code - Hybrid Image
    """

    print("Creating hybrid image...")
    output_path = 'images/outputs/hybrid_ver1.jpg'
    hybrid = hybrid_image(imgA_path, imgB_path, output_path,
                          low_pass_radius=15, high_pass_radius=15)

    print("\nDone ..!")
