"""Library for image hashing and deduplication."""

import hashlib

import numpy
from PIL import Image
import scipy.fftpack as fftpack


class Hash(object):
    """Object used for computing image hashes and testing duplicates."""
    def __init__(self, standard_width=128, edge_width=16, 
        key_frames=frozenset([0, 4, 9, 14, 19])):
        self._standard_width = standard_width
        self._edge_width = edge_width
        self._key_framss = sorted(list(key_frames))

    def hash_image(self, im):
        """Hash an image. Ignore details."""
        hasher = hashlib.md5()

        if _is_video(im):
            self._hash_video(im, hasher)
        else:
            self._hash_photo(im, hasher)

        return hasher.hexdigest()

    def test_duplicate(self, im1, im2):
        """Test whether two images are duplicates."""
        hash1 = self.hash_image(im1)
        hash2 = self.hash_image(im2)

        return hash1 == hash2

    def _hash_photo(self, im, hasher):
        # Mark the fact that this is in the images space.
        hasher.update('IMAGE')
        # Add the height of the image to the photo hash.
        _, height = im.size
        hasher.update('%d' % (height / 5))

        # Add the contents of the single frame to the hash.
        self._frame_hash(im, hasher)

    def _hash_video(self, im, hasher):
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
        def extract_coeff(coeff):
            return max(min(int(coeff), 1023), -1024) / 16
    
        im_gray = im.convert('L')
        (im_small, _) = _resize_to_width(im_gray, self._standard_width)
        mat = numpy.array(im_small, dtype=numpy.float) - 128
        ep = self._edge_width
        mat_core = mat[ep:(mat.shape[0]-ep), ep:(mat.shape[1]-ep)]
        mat_dct = fftpack.dct(fftpack.dct(mat_core, norm='ortho').T, norm='ortho').T
    
        for ii in range(0, 4):
            for jj in range(0, 4):
                hasher.update('%04d' % extract_coeff(mat_dct[ii][jj]))


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
