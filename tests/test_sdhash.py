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
    def test_hash_image_simple_run_one_coeff(self):
        md5hasher = _md5_sequence('IMAGE', '%d' % (32/5), '0125')

        hasher = sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=1)
        im = _build_test_image((32, 32), 0, [[1002]])

        self.assertEqual(hasher.hash_image(im), md5hasher.hexdigest())

    def test_hash_image_simple_run_four_coeffs(self):
        md5hasher = _md5_sequence('IMAGE', '%d' % (32/5), '0125', '0051', '0051', '0025')

        hasher = sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2)
        im = _build_test_image((32, 32), 0, [[1002, 412], [412, 206]])

        self.assertEqual(hasher.hash_image(im), md5hasher.hexdigest())


class ImageSyntheticLarge(TestCase):
    pass


class ImageReal(TestCase):
    pass


class AnimationSyntheticLarge(TestCase):
    pass


class AnimationReal(TestCase):
    pass


def _md5_sequence(*args):
    md5hasher = hashlib.md5()

    for arg in args:
        md5hasher.update(arg)

    return md5hasher


def _build_test_image(size, edge_width, core):
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


class TestSDHash(TestCase):
    def test_hash_image(self):
        hasher = sdhash.Hash(standard_width=8, edge_width=2)
        im = Image.new('RGB', (8, 8), 'black')
        self.assertEqual(hasher.hash_image(im), '9761febb046365e9ce0cf2f31e0918cc')

    def test_test_duplicate(self):
        hasher = sdhash.Hash()
        im1 = Image.new('RGB', (8, 8), 'black')
        im2 = Image.new('RGB', (8, 8), 'black')
        im3 = Image.new('RGB', (8, 8), 'white')
        self.assertTrue(hasher.test_duplicate(im1, im2))
        self.assertFalse(hasher.test_duplicate(im1, im3))


if __name__ == '__main__':
    unittest.main()
