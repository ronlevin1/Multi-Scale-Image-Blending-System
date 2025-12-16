# Multi-Scale Image Blending System

This project implements **Multi-Scale Image Blending** using **Gaussian and Laplacian pyramids**. The technique allows for seamless blending of two images by mixing their frequencies separately, preventing seams and ghosting artifacts common in direct composition.

Based on the method described by Burt and Adelson (1983).

## Features

* **Gaussian Pyramid:** Downsampling and smoothing for multi-scale representation.
* **Laplacian Pyramid:** Band-pass filtering to isolate image frequencies.
* **Seamless Blending:** Blends images at each pyramid level using a mask.
* **Reconstruction:** Rebuilds the high-resolution blended image from the pyramid.

## Files

* `main.py`: Entry point for running the blending script.
* `create_mask.py`: Contains a function for aoutomated mask creation based on face marks.
* `face_align.py`: Helper functions for image aligning by face marks.
* `inputs/`: Directory containing sample input images and masks.
* `outputs/`: Directory containing sample output images.

## Installation

Ensure you have Python installed along with the following dependencies:

```bash
pip install numpy matplotlib opencv-python
```

## Usage
Run the main script to blend two images. Ensure you have a source image, a target image, and a binary mask.

## Configuration
Adjust the parameters in main.py to point to your specific images:

* Image 1: Foreground/Source.

* Image 2: Background/Target.

* Mask: Binary mask defining the blending region.

* Levels: Depth of the pyramid (default is usually 5 or 6).

## Example Output
The system generates a blended image where the transition between the two source images is smooth, preserving the structure of both while matching lighting and texture gradients.

## References
* The Laplacian Pyramid as a Compact Image Code (Burt & Adelson, 1983)
