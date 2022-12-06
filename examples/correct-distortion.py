import cupy as cp
from imageio.v2 import imread, imwrite

from httomolib.correction import correct_distortion

# Load image to be corrected
file_path = "data/distortion-correction/dot_pattern_03.tif"
im_host = imread(file_path)
im = cp.asarray(im_host)

# Define the `preview` to not crop out any of the image
PREVIEW = {
    'starts': [0, 0],
    'stops': [im.shape[0], im.shape[1]],
    'steps': [1, 1]
}

# Point to the file containing the distortion coefficients (assumed to be
# calculated in advance of the main processing pipeline)
distortion_coeffs_file_path = 'data/distortion-correction/distortion-coeffs.txt'

# Apply distortion correction
corrected_images = \
    correct_distortion(im, distortion_coeffs_file_path, PREVIEW)
corrected_images = cp.squeeze(corrected_images)

# Save corrected image if desired
#out_file_path = 'corrected-distortion.tif'
#imwrite(out_file_path, im.get())
