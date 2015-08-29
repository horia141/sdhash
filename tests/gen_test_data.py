#!/usr/bin/env python
"""Generate a set of test data images from "originals".

This script should be run from the top level package directory, like this:

  >> ./tests/gen_test_data.py

It looks at images of the form ./tests/data/*.original.png and computes a number
of transforms on them. SDHash should be able to detect all these transforms as
still being the same image.

Depends on:
- The ImageMagick suite of programs
- The Python Image Library
"""

import glob
import os
import re
import subprocess

from PIL import Image 


_TEST_DATA_ORIGINAL = './tests/data/*.original.png'
_PREFIX_RE = re.compile('^(\w+).original.png$')
_GENERATORS = [
    {
        'name': 'Another copy of the original',
        'command': 'cp {0}.original.png {0}.originaldup.png',
        'outfile': '{0}.originaldup.png'
        },
    {
        'name': 'JPEG version with quality=95',
        'command': 'convert {0}.original.png -quality 95 {0}.qual95.jpg',
        'outfile': '{0}.qual95.jpg'
        },
    {
        'name': 'JPEG version with quality=80',
        'command': 'convert {0}.original.png -quality 80 {0}.qual80.jpg',
        'outfile': '{0}.qual80.jpg'
        },
    {
        'name': 'JPEG version with quality=70',
        'command': 'convert {0}.original.png -quality 75 {0}.qual75.jpg',
        'outfile': '{0}.qual75.jpg'
        },
    {
        'name': 'JPEG version with quality=50',
        'command': 'convert {0}.original.png -quality 50 {0}.qual50.jpg',
        'outfile': '{0}.qual50.jpg'
        },
    {
        'name': 'JPEG version with quality=25',
        'command': 'convert {0}.original.png -quality 25 {0}.qual25.jpg',
        'outfile': '{0}.qual25.jpg',
        'hasher': {'dct_core_width': 6, 'dct_coeff_buckets': 32},
        },
    {
        'name': 'GIF version',
        'command': 'convert {0}.original.png {0}.original.gif',
        'outfile': '{0}.original.gif',
        'hasher': {'dct_core_width': 6, 'dct_coeff_buckets': 32}
        },
    {
        'name': 'With gaussian noise at amplitude=0.1',
        'command': 'convert {0}.original.png -evaluate Gaussian-noise 0.1 {0}.noise01.png',
        'outfile': '{0}.noise01.png',
        'hasher': {'dct_coeff_buckets': 32}
        },
    {
        'name': 'With gaussian noise at amplitude=0.2',
        'command': 'convert {0}.original.png -evaluate Gaussian-noise 0.2 {0}.noise02.png',
        'outfile': '{0}.noise02.png',
        'hasher': {'dct_coeff_buckets': 16}
        },
    {
        'name': 'Scale 0.5x',
        'command': 'convert {0}.original.png -resize 50% {0}.scale050x.png',
        'outfile': '{0}.scale050x.png'
        },
    {
        'name': 'Scale 1.5x',
        'command': 'convert {0}.original.png -resize 150% {0}.scale150x.png',
        'outfile': '{0}.scale150x.png'
        },
    {
        'name': 'Scale 2x',
        'command': 'convert {0}.original.png -resize 200% {0}.scale200x.png',
        'outfile': '{0}.scale200x.png'
        },
    {
        'name': 'Scale 3x',
        'command': 'convert {0}.original.png -resize 300% {0}.scale300x.png',
        'outfile': '{0}.scale300x.png'
        },
    {
        'name': 'Grayscale version (via ImageMagick)',
        'command': 'convert {0}.original.png -colorspace Gray {0}.magickgray.png',
        'outfile': '{0}.magickgray.png',
        'hasher': {'dct_coeff_buckets': 4}
        },
    {
        'name': 'Grayscale version (via PIL)',
        'command': lambda k: _to_grayscale(k),
        'outfile': '{0}.pilgray.png'
        },
    ]

def _to_grayscale(kernel):
    x = Image.open('{0}.original.png'.format(kernel))
    y = x.convert('L')
    y.save('{0}.pilgray.png'.format(kernel))


def gen_test_data():
    test_cases = []

    for original in glob.glob(_TEST_DATA_ORIGINAL):
        base = os.path.basename(original)
        match = _PREFIX_RE.match(base)
        assert match is not None
        kernel = match.group(1)

        test_case = {
            'reference': base,
            'modified': []
            }

        os.chdir('./tests/data')
        try:
            for gen in _GENERATORS:
                if hasattr(gen['command'], '__call__'):
                    gen['command'](kernel)
                else:
                    subprocess.check_call(gen['command'].format(kernel), shell=True)
                test_case['modified'].append({
                    'name': gen['name'],
                    'image': gen['outfile'].format(kernel),
                    'hasher': gen.get('hasher', {}),
                    })
        except (subprocess.CalledProcessError, IOError) as e:
            print 'Error'
            print str(e)
            return
        finally:
            os.chdir('../..')

        test_cases.append(test_case)

    return test_cases


def clear_test_data(test_cases):
    for test_case in test_cases:
        os.chdir('./tests/data')
        for m in test_case['modified']:
            try:
                os.remove(m['image'])
            except Error as e:
                pass
        os.chdir('../..')


if __name__ == '__main__':
    gen_test_data()
