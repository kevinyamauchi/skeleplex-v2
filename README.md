# skeleplex-v2

[![License](https://img.shields.io/pypi/l/skeleplex-v2.svg?color=green)](https://github.com/kevinyamauchi/skeleplex-v2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/skeleplex-v2.svg?color=green)](https://pypi.org/project/skeleplex-v2)
[![Python Version](https://img.shields.io/pypi/pyversions/skeleplex-v2.svg?color=green)](https://python.org)
[![CI](https://github.com/kevinyamauchi/skeleplex-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinyamauchi/skeleplex-v2/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kevinyamauchi/skeleplex-v2/branch/main/graph/badge.svg)](https://codecov.io/gh/kevinyamauchi/skeleplex-v2)

## Work in progress

This is a work-in-progress re-write of SkelePlex. Currently, nothing is working. Please check back later!

## Install with dependencies for the viewer

```bash
pip install ".[cellier, viz]"
```


## Development installation

First, fork the library and clone your fork to your local computer. `cd` into the cloned directory.

```bash

cd path/to/skeleplex-v2
```

We recommend installinginto a fresh environment. Activate your skeleplex environment and then install in editable mode with the optional developer and testing dependencies

```bash

pip install -e ".[dev,test]"
```

Finally, install the pre-commit git hooks. These will apply the linting before you commit your changes.


```bash
pre-commit install
```

## Conventions

We transform all coordinates to physical units [µm].
The images the graphs are based on should be isotropic.

## Differences across OS

To compute signed distances to mesh surfaces, we use pysdf (https://github.com/sxyu/sdf) on UNIX-based systems and igl (https://libigl.github.io) on Windows. Both approaches are expected to yield consistent results, though pysdf generally offers better performance. This difference only affects the generation of synthetic data.