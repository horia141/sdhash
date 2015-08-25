"""Library for image hashing and deduplication."""

import hashlib
import math

import numpy
from PIL import Image
import scipy.fftpack as fftpack


class Hash(object):
    """Object used for computing image hashes and testing duplicates."""

    DCT_COEFF_MIN = -1024
    DCT_COEFF_MAX = 1023

    def __init__(self, standard_width=128, edge_width=16, key_frames=frozenset([0, 4, 9, 14, 19]),
            dct_core_width=4, dct_coeff_buckets=256):
        """Create a Hash object.

        Args:
          standard_width: the common width all images will be resized to. The height will be
            computed such that the aspect ratio is maintained.
          edge_width: how much of the resized image edges to discard. 
          key_frames: the set of frames to consider when computing hashes for animations.
          dct_core_width: the size of the top-left matrix of the DCT of the image to include in
            the hash computations.
          dct_coeff_buckets: the quantization level for DCT coefficients from the top-left matrix
            of the DCT of the image.
        """
        assert standard_width > 0
        assert edge_width >= 0
        assert edge_width <= standard_width / 2
        assert len(key_frames) > 0
        assert dct_core_width > 0
        assert dct_core_width <= standard_width - 2 * edge_width
        assert dct_coeff_buckets > 0

        self._standard_width = standard_width
        self._edge_width = edge_width
        self._key_frames = sorted(list(key_frames))
        self._dct_core_width = dct_core_width
        self._dct_coeff_buckets = dct_coeff_buckets
        self._dct_coeff_split = (self.DCT_COEFF_MAX - self.DCT_COEFF_MIN + 1) / dct_coeff_buckets
        self._lower_bound_fp_rate = 1.0 / (dct_core_width * dct_core_width * dct_coeff_buckets)

    def hash_image(self, im):
        """Hash an image. Ignore details.

        Args:
          im: a PIL image which will be hashed.

        Returns:
          An MD5 hash string, resistent to small small perceptual transformations.
        """
        hasher = hashlib.md5()

        if _is_video(im):
            self._hash_animation(im, hasher)
        else:
            self._hash_image(im, hasher)

        return hasher.hexdigest()

    def test_duplicate(self, im1, im2):
        """Test whether two images are duplicates.

        Args:
          im1: a PIL image.
          im2: a PIL image.

        Returns:
          Whether the two images are perceptually identical, according to hash_image.
        """
        hash1 = self.hash_image(im1)
        hash2 = self.hash_image(im2)

        return hash1 == hash2

    def _hash_image(self, im, hasher):
        # Mark the fact that this is in the images space.
        hasher.update('IMAGE')
        # Add the contents of the single frame to the hash.
        self._frame_hash(im, hasher)

    def _hash_animation(self, im, hasher):
        # Mark the fact that this is in the video space.
        hasher.update('VIDEO')
        # Add the height of the video to the photo hash.
        _, height = im.size
        hasher.update('%d' % (height / 5))

        # Add the contents of each key frame to the hash. Algorithm is kind of ugly.
        # We can only seek to consecutive frames. So this means we have to do the
        # sequence 0,1,2,... . Whenever we encounter a key frame, we hash it. We stop if
        # ther are no more frames in the video or no more key frames.
        frame_idx = 0
        key_frame_idx = 0
        while True:
            try:
                im.seek(frame_idx)
            except EOFError:
                break
            if frame_idx == self._keyframes[key_frame_idx]:
                self._frame_hash(im, hasher)
                key_frame_idx += 1
                if key_frame_idx >= len(self._keyframes):
                    break
            frame_idx += 1
        im.seek(0)

    def _frame_hash(self, im, hasher):
        im_gray = im.convert('F')
        (im_small, _) = _resize_to_width(im_gray, self._standard_width)
        mat = numpy.asarray(im_small, dtype=numpy.float32) - 128
        edge_width = self._edge_width
        mat_core = mat[edge_width:(mat.shape[0]-edge_width), edge_width:(mat.shape[1]-edge_width)]
        mat_dct = fftpack.dct(fftpack.dct(mat_core, norm='ortho').T, norm='ortho').T
    
        _, height_small = im_small.size
        hasher.update('%d' % (height_small / 5))

        for ii in range(0, self._dct_core_width):
            for jj in range(0, self._dct_core_width):
                hasher.update(self._prepare_coeff(mat_dct[ii][jj]))

    def _prepare_coeff(self, coeff):
        clamped = max(min(int(coeff), self.DCT_COEFF_MAX), self.DCT_COEFF_MIN) / self._dct_coeff_split
        sign = '+' if clamped > 0 else '-'
        return '%s%04d' % (sign, abs(clamped))

    @property
    def standard_width(self):
        return self._standard_width

    @property
    def edge_width(self):
        return self._edge_width

    @property
    def key_frames(self):
        return list(self._key_frames)

    @property
    def dct_core_width(self):
        return self._dct_core_width

    @property
    def dct_coeff_buckets(self):
        return self._dct_coeff_buckets

    @property
    def dct_coeff_split(self):
        return self._dct_coeff_split

    @property
    def lower_bound_fp_rate(self):
        return self._lower_bound_fp_rate
    

def _is_video(im):
    try:
        im.seek(1)
    except EOFError:
        return False
    im.seek(0)
    return True


def _resize_to_width(im, desired_width):
    (width, height) = im.size
    aspect_ratio = float(height) / float(width)
    desired_height = int(aspect_ratio * desired_width)
    desired_height = desired_height + desired_height % 2 # Always a multiple of 2.
    im_resized = im.resize((desired_width, desired_height), Image.ANTIALIAS)
    return (im_resized, desired_height)
