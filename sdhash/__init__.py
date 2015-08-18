"""Library for image hashing and deduplication."""

class Hash(object):
    """Object used for computing image hashes and testing duplicates."""
    def __init__(self):
        pass

    def hash_image(self, im):
        """Hash an image. Ignore details."""
        return 0

    def test_duplicate(self, im1, im2):
        """Test whether two images are duplicates."""
        return False
