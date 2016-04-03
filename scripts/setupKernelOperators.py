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


def cOperatorDeclarations(N):
    return '\n\n'.join(cOperatorDeclaration(n + 1) for n in range(N))

def cOperatorDeclaration(N):
    return '    OCCA_LFUNC void OCCA_RFUNC occaKernelRun{0}(occaKernel kernel, {1});\n'.format(N, ' '.join('void *arg' + str(n) + nlc(n, N) for n in range(N)) )


def cOperatorDefinitions(N):
    return '\n\n'.join(cOperatorDefinition(n + 1) for n in range(N))

def cOperatorDefinition(N):
    argsContent = ', '.join('((occaType) arg{})->ptr'.format(n) for n in range(N))

    return ('    void OCCA_RFUNC occaKernelRun{0}(occaKernel kernel, {1}){{\n'.format(N, ' '.join('void *arg' + str(n) + nlc(n, N) for n in range(N)) ) + \
            '      occa::kernel kernel_((occa::kernel_v*) kernel);\n'             + \
            '      kernel_.clearArgumentList();\n'                                + \
            '      \n'                                                            + \
            '      occaType_t *args[' + str(N) + '] = {' + argsContent + '};\n'   + \
            '      \n'                                                            + \
            '      for(int i = 0; i < ' + str(N) + '; ++i){\n'                    + \
            '        occaType_t &arg = *(args[i]);\n'                             + \
            '        void *argPtr    = arg.value.data.void_;\n'                   + \
            '      \n'                                                            + \
            '        if(arg.type == OCCA_TYPE_MEMORY){\n'                         + \
            '          occa::memory memory_((occa::memory_v*) argPtr);\n'         + \
            '          kernel_.addArgument(i, occa::kernelArg(memory_));\n'       + \
            '        }\n'                                                         + \
            '        else if(arg.type == OCCA_TYPE_PTR){\n'                       + \
            '          occa::memory memory_((void*) argPtr);\n'                   + \
            '          kernel_.addArgument(i, occa::kernelArg(memory_));\n'       + \
            '        }\n'                                                         + \
            '        else {\n'                                                    + \
            '          kernel_.addArgument(i, occa::kernelArg(arg.value));\n'     + \
            '          delete (occaType_t*) args[i];\n'                           + \
            '        }\n'                                                         + \
            '      }\n'                                                           + \
            '      \n'                                                            + \
            '      kernel_.runFromArguments();\n'                                 + \
            '    }\n');

occadir = osp.abspath(osp.join(osp.dirname(__file__), ".."))

def gen_file(filename, content):
    with open(occadir + filename, 'w') as f:
        f.write(content + '\n')

gen_file('/src/operators/runFunctionFromArguments.cpp' , runFunctionFromArguments(maxN))
gen_file('/include/occa/operators/declarations.hpp'    , operatorDeclarations(maxN))
gen_file('/src/operators/definitions.cpp'              , operatorDefinitions(maxN))
gen_file('/include/occa/operators/cKernelOperators.hpp', cOperatorDeclarations(maxN))
gen_file('/src/operators/cKernelOperators.cpp'         , cOperatorDefinitions(maxN))
