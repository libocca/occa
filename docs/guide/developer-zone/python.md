# Python

## Get source

```bash
git clone git@github.com:libocca/occa.py.git
cd occa.py
git submodule update --init
```

## Setting up environment

Create a `conda` environment or `virtualenv`

```bash
conda create -n py36 python=3.6
. activate py36
```

Setup a development environment with `pip`
```bash
pip install -e .
```

## Editing C++ modules

To recompile only the updated modules

```bash
# To avoid doing a `make clean` each time, use the `--no-clean` flag
python setup.py build_ext --no-clean --inplace
```
