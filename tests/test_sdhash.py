from unittest import TestCase

import sdhash


class TestSDHash(TestCase):
    def test_hash_image(self):
        hasher = sdhash.Hash()
        self.assertEqual(hasher.hash_image(10), 0)

    def test_test_duplicate(self):
        hasher = sdhash.Hash()
        self.assertFalse(hasher.test_duplicate(10, 20))


if __name__ == '__main__':
    unittest.main()
