# SDHash [![Build Status](https://travis-ci.org/horia141/sdhash.svg?branch=master)](https://travis-ci.org/horia141/sdhash)

A Python library for computing hashes of images which ignore perceptual differences.

## Usage

```python
import sdhash
from PIL import Image

i1 = Image.open('test1.png')
i2 = Image.open('test1_noise.png')
i3 = Image.open('test2.png')

h = sdhash.Hash()

h.test_duplicate(i1, i2) # True
h.test_duplicate(i1, i3) # False
h.hash_image(i1) # [ an md5 output ]
```

## Background

As humans, it's very easy to spot if two images are "the same". Unfortunately, the same
thing can't be said of computers. A simple approach such as comparing two images pixel
by pixel will fail in all but the most simplest of cases.

For example, given an original image A, the following should produce equivalent images:
* Scaling with the same aspect ratio.
* Scaling with a very similar aspect ratio.
* Adding high frequency noise.
* Small blurring / high pass filtering.
* Lossy compression and reconstruction.

The previous set of transformations can be said to be benign. There's an even bigger
class of transformations which are adversarial - somebody is trying to trick the system
into believing one image is original when it is in fact not. User generated content sites
face this problem, regardless of the mode of the content (text, image, audio, video).

The following adverserial transformations should produce equivalent images:
* Removing a small area around borders.
* 90, 180, 270 degrees rotation.
* Horizontal or vertical flipping.
* Adding or removing a watermark.
* Alteration of color planes, but not of the luminance one.

SDHash the problem of identifying if two images are identical or not, modulo all the
transformations in the first group, and some (removing of borders and color plane
alterations) from the second.

The API it exposes is simple. The `test_duplicate` method receives two PIL images as
input and returns either `True` or `False` depending on whether it considers the
images as equivalent or not. The `hash_image` method returns a base64 encoded md5
hash of "stable" image contents. The `test_duplicate` method is essentially a test
of whether the hash of the arguments is equal. For more advanced usage, the second
method is required. For example, a database table of the hashes can be used, with
the result of `hash_image` as a primary key. Whenever new image needs to be added it
can be checked first against the table and only if it is not found already, inserted.

## Algorithm

The core algorithm is straightforward:
* The input image is converted to a single plane of high-precision grayscale.
* This plane is resized to a standard width, while maintaing the aspect ratio.
* The DCT is computed on the plane.
* An MD5 hash is computed from a stream which starts with the width and height of the
image, and continues with the top-left most significant DCT coefficients, in row major
order, clamped to an interval and precision.

All details of the process can be controlled via arguments to the hasher object
constructor, although their effect is somewhat esoteric. Good defaults have been
provided.

## Installation

The Python image library and NumPy/SciPy etc.

Installation is simple, via `pip`:

```bash
pip install sdhash
```
