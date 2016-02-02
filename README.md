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
by pixel will fail in all but the most simple of cases. Even a more sophisticated method
such as computing an `L1` or `L2` mean distance between the images and taking a small
value to imply similarity is not much better.

For example, given an original image A, the following should produce equivalent images:
* Scaling with the same aspect ratio.
* Scaling with a very similar aspect ratio.
* Adding high frequency noise.
* Small blurring / high pass filtering.
* Lossy compression and reconstruction.

The previous set of transformations can be said to be __natural__, that is, they will most
certainly occur in any systen, as images get moved around, edited etc .
However, there's an even bigger class of transformations which are __adversarial__, that is,
somebody is trying to trick the system to believe one image is original when it is in
fact not. User generated content sites face this problem, regardless of the mode of the
content (text, image, audio, video).

The following adverserial transformations should produce equivalent images, as well:
* Removing a small area from the borders.
* 90, 180, 270 degrees rotation.
* Horizontal or vertical flipping.
* Adding or removing a watermark.
* Alteration of color planes, but not of the luminance one.
* Grayscale conversion.
* Large-scale editing restricted to a small area (replacing some text with another,
for example)

SDHash tries to solve the problem of whether two images are identical or not, modulo
all the transformations in the first group, and some (removing of borders and 
color plane lterations) from the second.

The API it exposes is simple. The `test_duplicate` method receives two PIL images as
input and returns either `True` or `False` depending on whether it considers the
images as equivalent or not. The `hash_image` method returns a base64 encoded md5
hash of "stable" image contents. The `test_duplicate` method is essentially a test
of whether the hashes of the arguments are equal. For more advanced usage, the second
method is the tool of choice.

For example, a database table of the hashes can be used, with the result of `hash_image`
as a primary key. Whenever new image needs to be added it can be checked first against
the table and only if it is not found already, inserted. This allows `O(1)` comparisons
to be performed for each insertion, instead of `O(n)`. Similarly, a MapReduce job can
compute the hashes of images in the map stage, which will result in all identical images
being grouped together with the same key in the reduce stage. This allows an `O(n)`
algorithm for deuplicating a large dataset.

As a bonus, SDHash works with GIF animations. It treats them as a sequence of frames.
Only the first, fifth, tenth etc. frames are considered. The same basic approach is used,
but all the frames are considered at once.

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

The main dependencies are on the Python image library and NumPy/SciPy etc. As such,
depending on SDHash brings along a lot of baggage. Ideally and since it only depends
on a very small piece of functionality, the library would just include the DCT code
directly. Feel free to contribute a well tested implementation.

Installation is simple, via `pip`:

```bash
pip install sdhash
```
