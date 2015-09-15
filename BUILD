py_library(
    name = "sdhash",
    srcs = ["sdhash/__init__.py"],
    srcs_version = "PY2"
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
