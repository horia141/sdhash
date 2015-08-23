# SDHash

A Python library for computing hashes of images which ignore perceptual differences.

## Usage

```python
import sdhash
from PIL import Image

i1 = Image.open('test1.png')
i2 = Image.open('test1_noise.png')
i3 = Image.open('test2.png')

sdhash.test_duplicate(i1, i2) # True
sdhash.test_duplicate(i1, i3) # False
sdhash.hash_image(i1) # [ an md5 output ]
```

## Background

Suppose you want to test that two images are identical. The naive approach of simply comparing the byte-array representation of the two is not good.

## Algorithm

## Dependencies

The Python image library and NumPy/SciPy etc.

## TODO

Resistance to rotation, mirroring etc.
Tunable knobs (for similarity detection etc.)
