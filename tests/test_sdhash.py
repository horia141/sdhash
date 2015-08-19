import hashlib
from unittest import TestCase

from PIL import Image

import sdhash


class TestSDHash(TestCase):
    def test_hash_image(self):
        hasher = sdhash.Hash(standard_width=8, edge_width=2)
        im = Image.new('RGB', (8, 8), 'black')
        self.assertEqual(hasher.hash_image(im), '5007cca1dbbc018703efc3e118281c63')

    def test_test_duplicate(self):
        hasher = sdhash.Hash()
        im1 = Image.new('RGB', (8, 8), 'black')
        im2 = Image.new('RGB', (8, 8), 'black')
        im3 = Image.new('RGB', (8, 8), 'white')
        self.assertTrue(hasher.test_duplicate(im1, im2))
        self.assertFalse(hasher.test_duplicate(im1, im3))


if __name__ == '__main__':
    unittest.main()
