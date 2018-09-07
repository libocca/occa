# The MIT License (MIT)
#
# Copyright (c) 2014-2018 David Medina and Tim Warburton
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import functools


OCCA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

EDIT_WARNING = '''
// -------------[ DO NOT EDIT ]-------------
//  THIS IS AN AUTOMATICALLY GENERATED FILE
//  EDIT: scripts/setup_kernel_operators.py
// =========================================
'''

MAX_ARGS = 50


def to_file(filename):
    def inner_to_file(func):
        @functools.wraps(func)
        def cached_func(*args, **kwargs):
            with open(OCCA_DIR + '/' + filename, 'w') as f:
                content = func(*args, **kwargs)
                f.write(EDIT_WARNING);
                f.write(content + '\n')
        return cached_func
    return inner_to_file


def operator_args(N, indent, argtype):
    content = ''
    for n in range(1, N + 1):
        content += '{argtype}arg{n}'.format(argtype=argtype,
                                            n=n)
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


@to_file('src/tools/runFunction.cpp')
def runFunctionFromArguments(N):
    content = '\nswitch (argc) {\n'
    for n in range(N + 1):
        content += runFunctionFromArgument(n)
    content += '}\n';

    return content


def runFunctionFromArgument(N):
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


@to_file('include/occa/core/kernelOperators.hpp')
def operatorDeclarations(N):
    return '\n\n'.join(
        operatorDeclaration(n) for n in range(N + 1)
    )


def operatorDeclaration(N):
    content = 'void operator () ('
    indent = ' ' * len(content)
    content += operator_args(N, indent, 'const kernelArg &')
    content += ') const;'

    return content


@to_file('src/core/kernelOperators.cpp')
def operatorDefinitions(N):
    return '\n'.join(
        operatorDefinition(n) for n in range(N + 1)
    )


def operatorDefinition(N):
    content = 'void kernel::operator() ('
    indent = ' ' * len(content)
    content += operator_args(N, indent, 'const kernelArg &')
    if N > 0:
        content += ''') const {{
  kernelArg args[] = {{
    {array_args}
  }};
  modeKernel->arguments.clear();
  modeKernel->arguments.reserve({N});
  modeKernel->arguments.insert(modeKernel->arguments.begin(), args, args + {N});
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
    runFunctionFromArguments(MAX_ARGS)
    operatorDeclarations(MAX_ARGS)
    operatorDefinitions(MAX_ARGS)
