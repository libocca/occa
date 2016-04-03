import os.path as osp

maxN = 50
nSpacing = 3

def nlc(n, N):
    ret = ''
    if n < (N - 1):
        ret = ', '
    if n != (N - 1) and ((n + 1) % nSpacing) == 0:
        ret += '\n                     '
    return ret;

def runFunctionFromArguments(N):
    return 'switch(argc){\n' + '\n'.join(runFunctionFromArgument(n + 1) for n in range(N)) + '}'

def runFunctionFromArgument(N):
    return '  case ' + str(N) + """:
    f(occaKernelInfoArgs, occaInnerId0, occaInnerId1, occaInnerId2, """ + ', '.join('args[{0}]'.format(n) for n in range(N)) + """); break;"""


def operatorDeclarations(N):
    return '\n\n'.join(operatorDeclaration(n + 1) for n in range(N))

def operatorDeclaration(N):
    return '    void operator () ({0});'.format( ' '.join('const kernelArg &arg' + str(n) + nlc(n, N) for n in range(N)) )


def operatorDefinitions(N):
    return '\n\n'.join(operatorDefinition(n + 1) for n in range(N))

def operatorDefinition(N):
    return """  void kernel::operator() (""" + ' '.join('const kernelArg &arg' + str(n) + nlc(n, N) for n in range(N)) + """){
    kernelArg args[] = {""" + ', '.join('arg{}'.format(n) for n in range(N)) + """}};
    kHandle->arguments.clear();
    kHandle->arguments.reserve({0});
    kHandle->arguments.insert(kHandle->arguments.begin(), args, args + {0});
    runFromArguments();
  }}""".format(N)


def cKernelDeclarations(N):
    return '\n\n'.join(cKernelDeclaration(n + 1) for n in range(N))

def cKernelDeclaration(N):
    return 'OCCA_LFUNC void OCCA_RFUNC occaKernelRun{0}(occaKernel kernel, {1});'.format(N, ' '.join('void *arg' + str(n) + nlc(n, N) for n in range(N)) )


def cKernelDefinitions(N):
    return '\n\n'.join(cKernelDefinition(n + 1) for n in range(N))

def cKernelDefinition(N):
    argsContent = ', '.join('((occaType) arg{})->ptr'.format(n) for n in range(N))

    return ('void OCCA_RFUNC occaKernelRun{0}(occaKernel kernel, {1}){{\n'.format(N, ' '.join('void *arg' + str(n) + nlc(n, N) for n in range(N)) ) + """
  occaType_t *args[{0}] = {{ {1} }};
  occaKernelRunN(kernel, {0}, args);
}}""".format(N, argsContent))

occadir = osp.abspath(osp.join(osp.dirname(__file__), ".."))

def gen_file(filename, content):
    with open(occadir + filename, 'w') as f:
        f.write(content + '\n')

gen_file('/src/operators/runFunctionFromArguments.cpp' , runFunctionFromArguments(maxN))
gen_file('/include/occa/operators/declarations.hpp'    , operatorDeclarations(maxN))
gen_file('/src/operators/definitions.cpp'              , operatorDefinitions(maxN))
gen_file('/include/occa/operators/cKernelOperators.hpp', cKernelDeclarations(maxN))
gen_file('/src/operators/cKernelOperators.cpp'         , cKernelDefinitions(maxN))
