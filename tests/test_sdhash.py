import hashlib
from unittest import TestCase

import numpy
from scipy import fftpack
from PIL import Image

import sdhash

# TODO(horia141):
#
# Image Synthetic Small-scale
# - small differences in DCT coeffs don't matter
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
            # Yup, it looks wierd. But we need to generate stuff in transpose space so
            # Image.formarray picks it up correctly. So we generate the coefficients in a transposd
            # form, and they'll be corrected when doing the final transpose.
            mat_dct[col_idx, row_idx] = value
            col_idx += 1
        row_idx += 1

    mat_core = fftpack.idct(fftpack.idct(mat_dct, norm='ortho').T, norm='ortho').T
    mat = numpy.float32(numpy.random.randint(0, 127, size))
    mat[edge_width:size[0] - edge_width, edge_width:size[1] - edge_width] = mat_core
    mat += 128

    return Image.fromarray(numpy.float32(mat).T, mode='F')


def _build_random_color_image(size):
    mat = numpy.uint8(numpy.random.randint(0, 255, (size[0], size[1], 3)))
    return Image.fromarray(mat, mode='RGB')


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
    HASH_IMAGE_TEST_CASES = [
        {
            'name': 'Simple run with one coeff',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=1),
            'image': _build_test_image((32, 32), 0, [[1002]]),
            'sequence': ['%d' % (32/5), '+0125']
            },
        {
            'name': 'Simple run with four coeffs',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'image': _build_test_image((32, 32), 0, [[1002, 412], [412, 206]]),
            'sequence': ['%d' % (32/5), '+0125', '+0051', '+0051', '+0025']
            },
        {
            'name': 'Simple run with four coeffs, but slightly different',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'image': _build_test_image((32, 32), 0, [[1003, 411], [413, 205]]),
            'sequence': ['%d' % (32/5), '+0125', '+0051', '+0051', '+0025']
            },
        {
            'name': 'Simple run with four coeffs, with one negative coeff',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'image': _build_test_image((32, 32), 0, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32/5), '+0125', '+0051', '-0027', '+0025']},
        {
            'name': 'Simple run with different number of buckets',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                                  dct_coeff_buckets=128),
            'image': _build_test_image((32, 32), 0, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32/5), '+0062', '+0025', '-0014', '+0012']
            },
        {
            'name': 'Coefficients get clamped',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'image': _build_test_image((32, 32), 0, [[2048, 412], [-1080, 206]]),
            'sequence': ['%d' % (32/5), '+0127', '+0051', '-0128', '+0025']
            },
        {
            'name': 'Only look at the core',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'image': _build_test_image((32, 32), 0, 
                [[1002, 412, 44], [-212, 206, -32], [33, 409, 23]]),
            'sequence': ['%d' % (32/5), '+0125', '+0051', '-0027', '+0025'],
            },
        {
            'name': 'Do not look at the edges',
            'hasher': sdhash.Hash(standard_width=32, edge_width=2, dct_core_width=2),
            'image': _build_test_image((32, 32), 2, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32/5), '+0125', '+0051', '-0027', '+0025'],
            },
        {
            'name': 'Do not look at the edges #2',
            'hasher': sdhash.Hash(standard_width=32, edge_width=4, dct_core_width=2),
            'image': _build_test_image((32, 32), 4, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32/5), '+0125', '+0051', '-0027', '+0025'],
            },
        {
            'name': 'Image gets shrunk (coeffs x0.5)',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'image': _build_test_image((64, 64), 0, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32/5), '+0062', '+0025', '-0014', '+0012'],
            },
        {
            'name': 'Image gets expanded (coeffs x2)',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'image': _build_test_image((16, 16), 0, [[501, 206], [-106, 103]]),
            'sequence': ['%d' % (32/5), '+0125', '+0051', '-0027', '+0025'],
            },
        {
            'name': 'Image gets shrunk, with kept aspect ratio (coeffs x0.25)',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'image': _build_test_image((64, 128), 0, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (64/5), '+0062', '+0025', '-0014', '+0012'],
            },
        ]

    def test_hash_image(self):
        for test_case in self.HASH_IMAGE_TEST_CASES:
            md5hasher = _md5_sequence('IMAGE', *test_case['sequence'])
            hash_code = test_case['hasher'].hash_image(test_case['image'])
            self.assertEqual(hash_code, md5hasher.hexdigest(),
                msg='Failed on "%s"' % test_case['name'])

    TEST_DUPLICATE_TEST_CASES = [
        {
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'reference': _build_test_image((32, 32), 0, [[1002, 412], [412, 206]]),
            'modified': [
                {
                    'name': 'Slightly different coeffs',
                    'image': _build_test_image((32, 32), 0, [[1003, 411], [413, 205]])
                    },
                {
                    'name': 'Only look at the core',
                    'image': _build_test_image((32, 32), 0, 
                        [[1002, 412, 44], [412, 206, -32], [33, 409, 23]])
                    },
                ]
            },
        {
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2),
            'reference': _build_test_image((32, 32), 0, [[1024, 412], [-1023, 206]]),
            'modified': [
                {
                    'name': 'Coefficients get clamped to reference',
                    'image': _build_test_image((32, 32), 0, [[2048, 412], [-1080, 206]]),
                    }
                ]
            },
        {
            'hasher': sdhash.Hash(standard_width=32, edge_width=2, dct_core_width=2),
            'reference': _build_test_image((32, 32), 2, [[1002, 412], [412, 206]]),
            'modified': [
                {
                    'name': 'Same border width, same coeffs',
                    'image': _build_test_image((32, 32), 2, [[1002, 412], [412, 206]])
                    },
                {
                    'name': 'Same border width, slightly different coeffs',
                    'image': _build_test_image((32, 32), 2, [[1003, 411], [413, 205]])
                    },
                ]
            },
        {
            'hasher': sdhash.Hash(standard_width=16, edge_width=0, dct_core_width=2),
            'reference': _build_test_image((16, 16), 0, [[1002, 412], [412, 206]]),
            'modified': [
                {
                    'name': 'Image of double size',
                    'image': _build_test_image((32, 32), 0, [[2004, 824], [824, 412]])
                    },
                {
                    'name': 'Image of half size',
                    'image': _build_test_image((8, 8), 0, [[501, 206], [206, 103]])
                    },
                ]
            },
        {
            'hasher': sdhash.Hash(standard_width=16, edge_width=0, dct_core_width=2),
            'reference': _build_test_image((16, 32), 0, [[1002, 412], [412, 206]]),
            'modified': [
                {
                    'name': 'Rectangular image of double size',
                    'image': _build_test_image((32, 64), 0, [[2004, 824], [824, 412]])
                    },
                {
                    'name': 'Rectangular image of half size',
                    'image': _build_test_image((8, 16), 0, [[501, 206], [206, 103]])
                    },
                ]
            },
        {
            'hasher': sdhash.Hash(standard_width=16, edge_width=0, dct_core_width=2),
            'reference': _build_random_color_image((16, 16)),
            'modified': [
                {
                    'name': 'Color components get ingored',
                    'image': lambda ref: ref.convert('F')
                    }
                ]
            }
        ]

    def test_test_duplicate(self):
        for test_case in self.TEST_DUPLICATE_TEST_CASES:
            hasher = test_case['hasher']
            reference = test_case['reference']

            for modified in test_case['modified']:
                if hasattr(modified['image'], '__call__'):
                    image = modified['image'](reference)
                else:
                    image = modified['image']

                self.assertTrue(hasher.test_duplicate(reference, image),
                     msg='Failed on "%s"' % modified['name'])


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
