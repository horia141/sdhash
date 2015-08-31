import hashlib
import logging
import os
from unittest import TestCase
import tabletest
from tabletest import TableTestCase

import numpy
from scipy import fftpack
from PIL import Image

import sdhash
import tests.gen_test_data as gen_test_data


logging.basicConfig(level=logging.INFO)


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


def _flatten_for_test_duplicate(test_cases):
    new_test_cases = []

    for test_case in test_cases:
        for m in test_case['modified']:
            new_test_cases.append({
               'name': m['name'],
               'reference': test_case['reference'],
               'modified': m['image'],
               'hasher': test_case['hasher']
               })

    return new_test_cases


class Core(TableTestCase):
    def test_construction(self):
        hasher = sdhash.Hash(
            standard_width=256,
            edge_width=24,
            key_frames=[0, 4, 9],
            height_buckets=128,
            dct_core_width=8,
            dct_coeff_buckets=256)

        self.assertEquals(hasher.standard_width, 256)
        self.assertEquals(hasher.edge_width, 24)
        self.assertEquals(hasher.key_frames, [0, 4, 9])
        self.assertEquals(hasher.height_buckets, 128)
        self.assertEquals(hasher.height_split, 16)
        self.assertEquals(hasher.dct_core_width, 8)
        self.assertEquals(hasher.dct_coeff_buckets, 256)
        self.assertEquals(hasher.dct_coeff_split, 8)

    def test_defaults_have_changed(self):
        hasher = sdhash.Hash()

        self.assertEquals(hasher.standard_width, 128)
        self.assertEquals(hasher.edge_width, 16)
        self.assertEquals(hasher.key_frames, [0, 4, 9, 14, 19])
        self.assertEquals(hasher.height_buckets, 256)
        self.assertEquals(hasher.height_split, 8)
        self.assertEquals(hasher.dct_core_width, 4)
        self.assertEquals(hasher.dct_coeff_buckets, 128)
        self.assertEquals(hasher.dct_coeff_split, 16)

    LOWER_BOUND_FP_RATE_TEST_CASES = [
        ({'dct_core_width': 2, 'dct_coeff_buckets': 128}, 1.0 / (2 * 2 * 128)),
        ({'dct_core_width': 4, 'dct_coeff_buckets': 64}, 1.0 / (4 * 4 * 64)),
        ({'dct_core_width': 8, 'dct_coeff_buckets': 32}, 1.0 / (8 * 8 * 32)),
        ({'dct_core_width': 16, 'dct_coeff_buckets': 32}, 1.0 / (16 * 16 * 32))
        ]

    @tabletest.tabletest(LOWER_BOUND_FP_RATE_TEST_CASES)
    def test_lower_bound_fp_rate(self, test_case):
        (input, expected_output) = test_case
        hasher = sdhash.Hash(**input)
        self.assertEquals(hasher.lower_bound_fp_rate, expected_output)


class ImageSynthetic(TableTestCase):
    HASH_IMAGE_TEST_CASES = [
        {
            'name': 'Simple run with one coeff',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=1,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 32), 0, [[1002]]),
            'sequence': ['%d' % (32 / 8), '+0125']
            },
        {
            'name': 'Simple run with four coeffs',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 32), 0, [[1002, 412], [412, 206]]),
            'sequence': ['%d' % (32 / 8), '+0125', '+0051', '+0051', '+0025']
            },
        {
            'name': 'Simple run with four coeffs, but slightly different',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 32), 0, [[1003, 411], [413, 205]]),
            'sequence': ['%d' % (32 / 8), '+0125', '+0051', '+0051', '+0025']
            },
        {
            'name': 'Simple run with four coeffs, with one negative coeff',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 32), 0, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32 / 8), '+0125', '+0051', '-0026', '+0025']
            },
        {
            'name': 'Simple run with different number of buckets',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                 dct_coeff_buckets=128),
            'image': _build_test_image((32, 32), 0, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32 / 8), '+0062', '+0025', '-0013', '+0012']
            },
        {
            'name': 'Coefficients get clamped',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 32), 0, [[2048, 412], [-1080, 206]]),
            'sequence': ['%d' % (32 / 8), '+0127', '+0051', '-0128', '+0025']
            },
        {
            'name': 'Image gets size bucketed',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=1,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 34), 0, [[0]]),
            'sequence': ['%d' % (34 / 8), '+0000']
            },
        {
            'name': 'Image gets size bucketed',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, height_buckets=100,
                 dct_coeff_buckets=256, dct_core_width=1),
            'image': _build_test_image((32, 34), 0, [[0]]),
            'sequence': ['%d' % (34 / 20.48), '+0000']
            },
        {
            'name': 'Only look at the core',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 32), 0, 
                [[1002, 412, 44], [-212, 206, -32], [33, 409, 23]]),
            'sequence': ['%d' % (32 / 8), '+0125', '+0051', '-0026', '+0025'],
            },
        {
            'name': 'Do not look at the edges',
            'hasher': sdhash.Hash(standard_width=32, edge_width=2, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 32), 2, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32 / 8), '+0125', '+0051', '-0026', '+0025'],
            },
        {
            'name': 'Do not look at the edges #2',
            'hasher': sdhash.Hash(standard_width=32, edge_width=4, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 32), 4, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32 / 8), '+0125', '+0051', '-0026', '+0025'],
            },
        {
            'name': 'Image gets shrunk (coeffs x0.5)',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((64, 64), 0, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (32 / 8), '+0062', '+0025', '-0013', '+0012'],
            },
        {
            'name': 'Image gets expanded (coeffs x2)',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((16, 16), 0, [[501, 206], [-106, 103]]),
            'sequence': ['%d' % (32 / 8), '+0125', '+0051', '-0026', '+0025'],
            },
        {
            'name': 'Image gets shrunk, with kept aspect ratio (coeffs x0.25)',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=2,
                 dct_coeff_buckets=256),
            'image': _build_test_image((64, 128), 0, [[1002, 412], [-212, 206]]),
            'sequence': ['%d' % (64 / 8), '+0062', '+0025', '-0013', '+0012'],
            },
        {
            'name': 'Giant image gets clamped in height',
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=1,
                 dct_coeff_buckets=256),
            'image': _build_test_image((32, 4096), 0, [[0]]),
            'sequence': ['%d' % (2048 / 8), '+0000'],
            },
        ]

    @tabletest.tabletest(HASH_IMAGE_TEST_CASES)
    def test_hash_image(self, test_case):
        md5hasher = _md5_sequence('IMAGE', *test_case['sequence'])
        hash_code = test_case['hasher'].hash_image(test_case['image'])
        self.assertEqual(hash_code, md5hasher.hexdigest(),
            msg='Failed on "%s"' % test_case['name'])

    TEST_DUPLICATE_TEST_CASES = _flatten_for_test_duplicate([
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
            'reference': _build_test_image((32, 32), 0, [[1023, 412], [-1027, 206]]),
            'modified': [
                {
                    'name': 'Coefficients get clamped to reference',
                    'image': _build_test_image((32, 32), 0, [[2048, 412], [-1080, 206]]),
                    }
                ]
            },
        {
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=1),
            'reference': _build_test_image((32, 34), 0, [[0]]),
            'modified': [
                {
                    'name': 'Height gets assigned to the same bucket',
                    'image': _build_test_image((32, 36), 0, [[0]]),
                    }
                ]
            },
        {
            'hasher': sdhash.Hash(standard_width=32, edge_width=0, dct_core_width=1),
            'reference': _build_test_image((32, 2048), 0, [[0]]),
            'modified': [
                {
                    'name': 'Height gets clamped to MAX_WIDTH',
                    'image': _build_test_image((32, 4096), 0, [[0]]),
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
        ])

    @tabletest.tabletest(TEST_DUPLICATE_TEST_CASES)
    def test_test_duplicate(self, test_case):
        reference = test_case['reference']
        if hasattr(test_case['modified'], '__call__'):
            modified = test_case['modified'](reference)
        else:
            modified = test_case['modified']
        hasher = test_case['hasher']
        self.assertTrue(hasher.test_duplicate(reference, modified),
             msg='Failed on "%s"' % test_case['name'])


class ImageReal(TableTestCase):
    TEST_CASES = gen_test_data.gen_test_data()

    @classmethod
    def tearDownClass(cls):
        gen_test_data.clear_test_data(cls.TEST_CASES)

    @tabletest.tabletest(TEST_CASES)
    def test_test_duplicate(self, test_case):
        reference = Image.open(os.path.join('tests', 'data', test_case['reference']))
        modified = Image.open(os.path.join('tests', 'data', test_case['modified']))
        hasher = sdhash.Hash(**(test_case.get('hasher', {})))
        self.assertTrue(hasher.test_duplicate(reference, modified),
            msg='Failed on "%s"' % test_case['name'])


class AnimationReal(TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
