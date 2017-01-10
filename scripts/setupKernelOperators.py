# The MIT License (MIT)
#
# Copyright (c) 2014-2016 David Medina and Tim Warburton
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

import os.path as osp

occadir = osp.abspath(osp.join(osp.dirname(__file__), ".."))

EDIT_WARNING = """/*
-------------[ DO NOT EDIT ]-------------
 THIS IS AN AUTOMATICALLY GENERATED FILE
 EDIT: scripts/setupKernelOperators.py
=========================================
*/
"""

maxN = 250
nSpacing = 3

def nlc(n, N):
    ret = ''
    if n < (N - 1):
        ret = ', '
    if n != (N - 1) and ((n + 1) % nSpacing) == 0:
        ret += '\n                     '
    return ret;

def runFunctionFromArguments(N):
    return 'switch (argc) {{ \n{cases} }}'.format(
        cases='\n'.join(runFunctionFromArgument(n + 1) for n in range(N)),
    )

def runFunctionFromArgument(N):
    return """  case {N}:
    f(occaKernelInfoArgs, occaInnerId0, occaInnerId1, occaInnerId2, {args}); break;""".format(
        N=N,
        args=', '.join('args[{0}]'.format(n) for n in range(N)),
    )

def operatorDeclarations(N):
    return '\n\n'.join(operatorDeclaration(n + 1) for n in range(N))

def operatorDeclaration(N):
    return '    void operator () ({args});'.format(
        args=' '.join('const kernelArg &arg' + str(n) + nlc(n, N) for n in range(N)),
    )


def operatorDefinitions(N):
    return '\n\n'.join(operatorDefinition(n + 1) for n in range(N))

def operatorDefinition(N):
    return """  void kernel::operator() ({args}) {{
    kernelArg args[] = {{ {argarray} }};
    kHandle->arguments.clear();
    kHandle->arguments.reserve({N});
    kHandle->arguments.insert(kHandle->arguments.begin(), args, args + {N});
    runFromArguments();
  }}""".format(
      N=N,
      args=' '.join('const kernelArg &arg' + str(n) + nlc(n, N) for n in range(N)),
      argarray=', '.join('arg{n}'.format(n=n) for n in range(N)),
  )


def cKernelDeclarations(N):
    return '\n\n'.join(cKernelDeclaration(n + 1) for n in range(N))

def cKernelDeclaration(N):
    return 'OCCA_LFUNC void OCCA_RFUNC occaKernelRun{N}(occaKernel kernel, {args});'.format(
        N=N,
        args=' '.join('occaType arg' + str(n) + nlc(n, N) for n in range(N)),
    )

def cKernelDefinitions(N):
    return '\n\n'.join(cKernelDefinition(n + 1) for n in range(N))

def cKernelDefinition(N):
    return """
void OCCA_RFUNC occaKernelRun{N}(occaKernel kernel, {args}) {{
  occaType args[{N}] = {{ {argarray} }};
  occaKernelRunN(kernel, {N}, args);
}}""".format(
        N=N,
        args=' '.join('occaType arg' + str(n) + nlc(n, N) for n in range(N)),
        argarray=', '.join('arg{n}'.format(n=n) for n in range(N)),
    )

def gen_file(filename, content):
    with open(occadir + filename, 'w') as f:
        f.write(EDIT_WARNING);
        f.write(content + '\n')

gen_file('/src/operators/runFunctionFromArguments.cpp' , runFunctionFromArguments(maxN))
gen_file('/include/occa/operators/declarations.hpp'    , operatorDeclarations(maxN))
gen_file('/src/operators/definitions.cpp'              , operatorDefinitions(maxN))
gen_file('/include/occa/operators/cKernelOperators.hpp', cKernelDeclarations(maxN))
gen_file('/src/operators/cKernelOperators.cpp'         , cKernelDefinitions(maxN))
