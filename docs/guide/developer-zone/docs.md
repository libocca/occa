# Docs

## Downloading

To update the docs, clone the git repo first and go to the docs directory

```bash
git clone git@github.com:libocca/occa.git
cd occa
```

## Yarn

Install `docsify` through `yarn`

```bash
yarn global add docsify-cli
```

Add global npm modules to `$PATH`

```bash
PATH+=":${HOME}/.npm-global/bin/docsify"
```

## Run libocca.org Locally

```bash
docsify serve docs
```

## Generate the API

### Requirements

Generating the API docs requires:
- Python 3.7 or greater
- doxygen
  - `sudo apt install doxygen`
- lxml Python package
-   `pip install lxml`

### Command

Inside the `occa/` directory, run

```bash
./scripts/docs/api-docgen
```
