# Docs

## Downloading

To update the docs, clone the git repo first

```bash
git clone git@github.com:libocca/libocca.org.git
cd libocca.org
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
docsify serve .
```
