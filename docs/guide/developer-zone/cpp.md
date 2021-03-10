# C++

## Running tests

```bash
make test
```

## Running a specific test

Shortcut for compiling and running tests

```bash
function ot {
    file=${1/\.cpp/}
    make "${OCCA_DIR}/tests/bin/${file}" && "${OCCA_DIR}/tests/bin/${file}"
}

function _ot {
    local prefix=${COMP_WORDS[COMP_CWORD]}
    local use=$(cd tests/src && find -type f | sed 's,^\./\(.*\)\.cpp,\1,g')
    COMPREPLY=($(compgen -W "$use" -- "${COMP_WORDS[COMP_CWORD]}"))
}

complete -F _ot ot
```

The bash function `ot` has autocomplete

```bash
> ot la[TAB]
lang/builtins/transforms/finders  lang/keyword                      lang/mode/opencl                  lang/parser                       lang/stream                       lang/tokenizer/movement           lang/type
lang/expression                   lang/mode/cuda                    lang/mode/openmp                  lang/preprocessor                 lang/tokenContext                 lang/tokenizer/string
lang/file                         lang/mode/okl                     lang/mode/serial                  lang/primitive                    lang/tokenizer/misc               lang/tokenizer/token
> ot lang/mo[TAB]
lang/mode/cuda    lang/mode/okl     lang/mode/opencl  lang/mode/openmp  lang/mode/serial
> ot lang/mode/cu[TAB]
> ot lang/mode/cuda
```
