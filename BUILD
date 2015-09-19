load("/tools/pypi_package", "pypi_package")

py_library(
    name = "sdhash",
    srcs = ["sdhash/__init__.py"],
)

filegroup(
    name = "sdhash_test_data",
    srcs = glob(["tests/data/*.png"])
)

py_test(
    name = "sdhash_test",
    main = "tests/test_sdhash.py",
    srcs = [
      "tests/__init__.py",
      "tests/gen_test_data.py",
      "tests/test_sdhash.py"
    ],
    deps = [
      ":sdhash",
      "@tabletest//:tabletest",
    ],
    data = [
       ":sdhash_test_data"
    ],
    size = "medium",
    default_python_version = "PY2",
    srcs_version = "PY2",
)

pypi_package(
    name = "sdhash_pkg",
    version = "0.0.3",
    description = "Library for image hashing and deduplication.",
    long_description = "README.md",
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
    keywords = "photo image gif hash perceptual dedup deduplication fft",
    url = "http://github.com/horia141/sdhash",
    author = "Horia Coman",
    author_email = "horia141@gmail.com",
    license = "MIT",
    packages = [":sdhash"],
    install_requires = ["pillow", "numpy", "scipy"],
    test_suite = "nose.collector",
    tests_require = ["nose", "@tabletest//:tabletest_pkg"],
)
