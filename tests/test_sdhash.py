import hashlib
from unittest import TestCase

import numpy
from scipy import fftpack
from PIL import Image

import sdhash

# TODO(horia141):
# A plan for testing
# Must have a way of transforming DCT matrices into images
#
# Core
# + construction & property access
# + changes in defaults
# + lower bounds for FP rate
#
# Image Synthetic Small-scale
# - one simple run through the hash_image
# - coefficients which are too large are clamped
# - coefficients which are too small are clamped
# - small differences in DCT coeffs don't matter
# - only components in "core" matter
# - edge components are ignoded
# - images larger than MAX_HEIGHT are equivalent
# - color components don't matter
#
# Image Synthetic Large-scale
# - images at different resolutions are equivalent
# - adding some noise doesn't matter
# - some images are trully different though
#
# Images Real
# - images at different resolution
# - images with noise
# - different images
#
# Animation Synthetic Large-scale
#
# Animation Real


def _build_test_image(size, edge_width, core):
    """Build a synthetic test image from given DCT coefficients.

    The result is a PIL image of size[0] rows and size[1] columns, in the 'F' format. Core is a list
    of lists which describes the DCT coefficients to be placed at the core of the image. It must be
    rectangular (that is, all lists should have the same length), but not necessarily square. An
    initial (size[0]-2*edge_width)x(size[1]-2*edge_width) is generated from these DCT coefficients,
    and then embedded in the center of a black image.

    The output format is 'F' in order to ensure that Hash.hash_image gets as close to the
    coefficients in core as possible.

    Hint: ensure that the coefficients in core are not close to a multiple of Hash.dct_coeff_split,
    since the rounding used is always floor, and the FP errors might lead to a number such as 1000
    turninginto 999.99, which gets rounded to 999 and which is 124 when divided by the standard
    Hash.dct_coeff_split of 8, rather than the original 125.

    Args:
      size: a two-element sequence containing the number of rows and columns of the output image.
      edge_width: number of edge pixels in the image.
      core: a rectangular list-of-lists containing DCT coefficients used to generate the image.

    Returns:
      A PIL image as described above.
    """
    mat_dct = numpy.zeros((size[0] - 2 * edge_width, size[1] - 2 * edge_width))
    row_idx = 0
    for row in core:
        col_idx = 0
        for value in row:
            mat_dct[row_idx, col_idx] = value
            col_idx += 1
        row_idx += 1

    mat_core = fftpack.idct(fftpack.idct(mat_dct, norm='ortho').T, norm='ortho').T
    mat = numpy.zeros(size)
    mat[edge_width:size[0] - edge_width, edge_width:size[1] - edge_width] = mat_core
    mat += 128

    return Image.fromarray(numpy.float32(mat), mode='F')


def _md5_sequence(*args):
    md5hasher = hashlib.md5()

    for arg in args:
        md5hasher.update(arg)

    return md5hasher


class Core(TestCase):
    def test_construction(self):
        hasher = sdhash.Hash(
            standard_width=256,
            edge_width=24,
            key_frames=[0, 4, 9],
            dct_core_width=8,
            dct_coeff_buckets=128)

        self.assertEquals(hasher.standard_width, 256)
        self.assertEquals(hasher.edge_width, 24)
        self.assertEquals(hasher.key_frames, [0, 4, 9])
        self.assertEquals(hasher.dct_core_width, 8)
        self.assertEquals(hasher.dct_coeff_buckets, 128)
        self.assertEquals(hasher.dct_coeff_split, 16)

    def test_defaults_have_changed(self):
        hasher = sdhash.Hash()

        self.assertEquals(hasher.standard_width, 128)
        self.assertEquals(hasher.edge_width, 16)
        self.assertEquals(hasher.key_frames, [0, 4, 9, 14, 19])
        self.assertEquals(hasher.dct_core_width, 4)
        self.assertEquals(hasher.dct_coeff_buckets, 256)
        self.assertEquals(hasher.dct_coeff_split, 8)

    def test_lower_bound_fp_rate(self):
        TEST_CASES = [
            ({'dct_core_width': 2, 'dct_coeff_buckets': 128}, 1.0 / (2 * 2 * 128)),
            ({'dct_core_width': 4, 'dct_coeff_buckets': 64}, 1.0 / (4 * 4 * 64)),
            ({'dct_core_width': 8, 'dct_coeff_buckets': 32}, 1.0 / (8 * 8 * 32)),
            ({'dct_core_width': 16, 'dct_coeff_buckets': 32}, 1.0 / (16 * 16 * 32))
            ]

        for (input, expected_output) in TEST_CASES:
            hasher = sdhash.Hash(**input)
            self.assertEquals(hasher.lower_bound_fp_rate, expected_output)


class ImageSyntheticSmall(TestCase):
    TEST_CASES = [
        {'name': 'Simple run with one coeff',
         'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=1),
         'image': _build_test_image((32, 32), 0, [[1002]]),
         'sequence': ['%d' % (32/5), '+0125']},
        {'name': 'Simple run with four coeffs',
         'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
         'image': _build_test_image((32, 32), 0, [[1002, 412], [412, 206]]),
         'sequence': ['%d' % (32/5), '+0125', '+0051', '+0051', '+0025']}
        ]

    def test_hash_image(self):
        for test_case in self.TEST_CASES:
            md5hasher = _md5_sequence('IMAGE', *test_case['sequence'])
            hash_code = test_case['hasher'].hash_image(test_case['image'])
            self.assertEqual(hash_code, md5hasher.hexdigest(),
                msg='Failed on "%s"' % test_case['name'])


class ImageSyntheticLarge(TestCase):
    pass


class ImageReal(TestCase):
    pass


class AnimationSyntheticLarge(TestCase):
    pass


class AnimationReal(TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
