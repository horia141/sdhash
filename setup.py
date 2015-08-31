from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='sdhash',
    version='0.0.1',
    description='Library for image hashing and deduplication.',
    long_description=readme(),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Video',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='photo image gif hash perceptual dedup deduplication fft',
    url='http://github.com/horia141/sdhash',
    author='Horia Coman',
    author_email='horia141@gmail.com',
    license='MIT',
    packages=[
        'sdhash',
    ],
    install_requires=['pillow', 'numpy', 'scipy'],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose', 'tabletest'],
)
