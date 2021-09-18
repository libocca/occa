#! /usr/bin/env python3

"""
Generates:
- `operator ()` for occa::kernel
- `switch ()` to call function pointers with varying argument counts
"""

import os
import functools


OCCA_DIR = os.environ.get(
    'OCCA_DIR',
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

EDIT_WARNING = f'''
// -------------[ DO NOT EDIT ]-------------
//  THIS IS AN AUTOMATICALLY GENERATED FILE
//  EDIT: {os.path.relpath(__file__, OCCA_DIR)}
// =========================================
'''.strip()

MAX_ARGS = 50


def to_file(filename):
    def inner_to_file(func):
        @functools.wraps(func)
        def cached_func(*args, **kwargs):
            with open(OCCA_DIR + '/' + filename, 'w') as f:
                content = func(*args, **kwargs)
                f.write(EDIT_WARNING + '\n\n');
                f.write(content + '\n')
        return cached_func
    return inner_to_file


def operator_args(N, indent, argtype, argname=None):
    content = ''
    for n in range(1, N + 1):
        if callable(argtype):
            argtype_n = argtype(n)
        else:
            argtype_n = argtype

        if callable(argname):
            argname_n = argname(n)
        else:
            argname_n = argname or 'arg{}'.format(n)
        content += argtype_n + argname_n
        if n < N:
            if n % 5:
                content += ', '
            else:
                content += ',\n' + indent

    return content


def array_args(N, indent):
    content = ''
    for n in range(1, N + 1):
        content += 'arg{n}'.format(n=n)
        if n < N:
            if n % 10:
                content += ', '
            else:
                content += ',\n' + indent

    return content


@to_file('src/occa/internal/utils/runFunction.cpp_codegen')
def run_function_from_arguments(N):
    content = '\nswitch (argc) {\n'
    for n in range(N + 1):
        content += run_function_from_argument(n)
    content += '}\n';

    return content


def run_function_from_argument(N):
    content  = '  case {N}:\n'.format(N=N)
    content += '    f('
    indent = ' ' * 6  # '    f('

    for n in range(N):
        content += 'args[{n}]'.format(n=n)
        if n < (N - 1):
            if (n + 1) % 5:
                content += ', '
            else:
                content += ',\n' + indent
    content += ');\n    break;\n'

    return content


@to_file('include/occa/core/kernelOperators.hpp_codegen')
def operator_declarations(N):
    # We manually define the 0-argument kernel for documentation purposes
    return '\n\n'.join(
        operator_declaration(n + 1) for n in range(N)
    )


def operator_declaration(N):
    content = 'void operator () ('
    indent = ' ' * len(content)
    content += operator_args(N, indent, 'const kernelArg &')
    content += ') const;'

    return content


@to_file('src/core/kernelOperators.cpp_codegen')
def operator_definitions(N):
    return '\n'.join(
        operator_definition(n) for n in range(N + 1)
    )


def operator_definition(N):
    content = 'void kernel::operator() ('
    indent = ' ' * len(content)
    content += operator_args(N, indent, 'const kernelArg &')
    if N > 0:
        content += ''') const {{
  assertInitialized();
  kernelArg args[] = {{
    {array_args}
  }};
  modeKernel->setArguments(args, {N});
  run();
}}
'''.format(N=N,
           array_args=array_args(N, ' ' * 4))
    else:
        content += ''') const {
  modeKernel->arguments.clear();
  run();
}
'''
    return content

if __name__ == '__main__':
    run_function_from_arguments(MAX_ARGS)
    operator_declarations(MAX_ARGS)
    operator_definitions(MAX_ARGS)
