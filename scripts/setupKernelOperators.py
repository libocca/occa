from os import environ as ENV

maxN = 50
nSpacing = 3

def vnlc(n, N):
    ret = ''
    if n < (N - 1):
        ret = ', '

    if n != (N - 1) and ((n + 1) % nSpacing) == 0:
        ret += '\n                             '

    return ret;

def nlc(n, N):
    ret = ''
    if n < (N - 1):
        ret = ', '

    if n != (N - 1) and ((n + 1) % nSpacing) == 0:
        ret += '\n                     '

    return ret;

def runFunctionFromArguments(N):
    return 'switch(argc){\n' + '\n'.join([runFunctionFromArgument(n + 1) for n in range(N)]) + '}'

def runFunctionFromArgument(N):
    return '  case ' + str(N) + """:
    f(occaKernelInfoArgs, occaInnerId0, occaInnerId1, occaInnerId2, """ + ', '.join(['args[{0}]'.format(n) for n in range(N)]) + """); break;"""

def runKernelFromArguments(N):
    return 'switch(argc){\n' + '\n'.join([runKernelFromArgument(n + 1) for n in range(N)]) + '}'

def runKernelFromArgument(N):
    return '  case ' + str(N) + """:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(""" + ', '.join(['args[{0}]'.format(n) for n in range(N)]) + """);
  }""" + (("""
  else{
    (*kHandle)(kHandle->nestedKernels, """ + ', '.join(['args[{0}]'.format(n) for n in range(N)]) + """);
  }""") if (N < maxN) else '') + """
  break;"""

def virtualOperatorDeclarations(N):
    return '\n\n'.join([virtualOperatorDeclaration(n + 1) for n in range(N)])

def virtualOperatorDeclaration(N):
    return '    virtual void operator () ({0}) = 0;'.format( ' '.join(['const kernelArg &arg' + str(n) + vnlc(n, N) for n in range(N)]) )

def operatorDeclarations(mode, N):
    return '\n\n'.join([operatorDeclaration(mode, n + 1) for n in range(N)])

def operatorDeclaration(mode, N):
    if mode == 'Base':
        ret = '    void operator () ({0});'.format( ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in range(N)]) )
    else:
        ret = '    template <>\n'\
              + '    void kernel_t<{0}>::operator () ({1});'.format(mode, ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in range(N)]) )

    return ret

def operatorDefinitions(mode, N):
    return '\n\n'.join([operatorDefinition(mode, n + 1) for n in range(N)])

def operatorDefinition(mode, N):
    if mode == 'Base':
        return """  void kernel::operator() (""" + ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in range(N)]) + """){
    """ + '\n    '.join(['arg' + str(n) + '.setupForKernelCall(kHandle->metaInfo.argIsConst(' + str(n) + '));' for n in range(N)]) + """

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(""" + ' '.join(['arg' + str(n) + nlc(n, N) for n in range(N)]) + """);
    }
    else{""" + (("""
      (*kHandle)(kHandle->nestedKernels, """ + ' '.join(['arg' + str(n) + nlc(n, N) for n in range(N)]) + """);""") \
                if (N < maxN) else '') + """
    }
  }

  void kernelDatabase::operator() (""" + ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in range(N)]) + """){/*
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    """ + '    '.join([('else if(arg' + str(n + 1) + '.dHandle) launchDevice = const_cast<occa::device_v*>(arg' + str(n + 1) + '.dHandle);\n') for n in range(N - 1)]) + """
    (*this)[launchDevice](""" + ' '.join(['arg' + str(n) + nlc(n, N) for n in range(N)]) + """);*/
  }"""
    else:
        header = operatorDefinitionHeader(mode, N)
        return header + operatorModeDefinition[mode](N) + "\n  }"

def operatorDefinitionHeader(mode, N):
    return """  template <>
  void kernel_t<{0}>::operator () ({1}){{""".format(mode, ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in range(N)]))

def pthreadOperatorDefinition(N):
    return """
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[""" + str(N) + """] = {""" + ' '.join(['arg' + str(n) + nlc(n, N) for n in range(N)]) + """};

    pthreads::runFromArguments(data_, dims, inner, outer, """ + str(N) + """, args);"""

def serialOperatorDefinition(N):
    return """
    SerialKernelData_t &data_ = *((SerialKernelData_t*) data);
    handleFunction_t tmpKernel = (handleFunction_t) data_.handle;
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    int argc = 0;

    const kernelArg *args[""" + str(N) + """] = {""" + ' '.join(['&arg' + str(n) + nlc(n, N) for n in range(N)]) + """};

    for(int i = 0; i < """ + str(N) + """; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    cpu::runFunction(tmpKernel,
                     occaKernelArgs,
                     occaInnerId0, occaInnerId1, occaInnerId2,
                     argc, data_.vArgs);"""

def ompOperatorDefinition(N):
    return """
    OpenMPKernelData_t &data_ = *((OpenMPKernelData_t*) data);
    handleFunction_t tmpKernel = (handleFunction_t) data_.handle;
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    int argc = 0;

    const kernelArg *args[""" + str(N) + """] = {""" + ' '.join(['&arg' + str(n) + nlc(n, N) for n in range(N)]) + """};

    for(int i = 0; i < """ + str(N) + """; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    cpu::runFunction(tmpKernel,
                     occaKernelArgs,
                     occaInnerId0, occaInnerId1, occaInnerId2,
                     argc, data_.vArgs);"""

def clOperatorDefinition(N):
    return """
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argc = 0;

    const kernelArg *args[""" + str(N) + """] = {""" + (', '.join((('\n                                 ' + (' ' if (10 <= N) else ''))
                                                                   if (n and ((n % 5) == 0))
                                                                   else '')
                                                                  + "&arg{0}".format(n) for n in range(N))) + """};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argc++, sizeof(void*), NULL));

    for(int i = 0; i < """ + str(N) + """; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argc++, args[i]->args[j].size, args[i]->args[j].ptr()));
      }
    }

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));"""

def cudaOperatorDefinition(N):
    return """
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[""" + str(N) + """] = {""" + (', '.join((('\n                                 ' + (' ' if (10 <= N) else ''))
                                                                   if (n and ((n % 5) == 0))
                                                                   else '')
                                                                  + "&arg{0}".format(n) for n in range(N))) + """};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < """ + str(N) + """; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));"""

def coiOperatorDefinition(N):
    return """
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + """ + str(N) + """*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    """ + '\n    '.join(["""if(arg{0}.pointer){{
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg{0}.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }}
    else{{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg{0}.arg), arg{0}.size);
      hostPos += arg{0}.size;
    }}""".format(n) for n in range(N)]) + """

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[""" + str(N - 1) + """],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));"""


# TW: needs to be adapted for HSA args stuff
def hsaOperatorDefinition(N):
    return """
    HSAKernelData_t &data_ = *((HSAKernelData_t*) data);
    HSAfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[""" + str(N) + """] = {""" + (', '.join((('\n                                 ' + (' ' if (10 <= N) else ''))
                                                                   if (n and ((n % 5) == 0))
                                                                   else '')
                                                                  + "&arg{0}".format(n) for n in range(N))) + """};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < """ + str(N) + """; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_HSA_CHECK("Launching Kernel",
                    hsaLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((HSAstream*) dHandle->currentStream),
                                   data_.vArgs, 0));"""


def cOperatorDeclarations(N):
    return '\n\n'.join([cOperatorDeclaration(n + 1) for n in range(N)])

def cOperatorDeclaration(N):
    return '    OCCA_LFUNC void OCCA_RFUNC occaKernelRun{0}(occaKernel kernel, {1});\n'.format(N, ' '.join(['void *arg' + str(n) + nlc(n, N) for n in range(N)]) )

def cOperatorDefinitions(N):
    return '\n\n'.join([cOperatorDefinition(n + 1) for n in range(N)])

def cOperatorDefinition(N):
    argsContent = ', '.join('((occaType) arg{})->ptr'.format(n) for n in range(N))

    return ('    void OCCA_RFUNC occaKernelRun{0}(occaKernel kernel, {1}){{\n'.format(N, ' '.join(['void *arg' + str(n) + nlc(n, N) for n in range(N)]) ) + \
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

# Removed COI
operatorModeDefinition = { 'Serial'   : serialOperatorDefinition,
                           'OpenMP'   : ompOperatorDefinition,
                           'OpenCL'   : clOperatorDefinition,
                           'CUDA'     : cudaOperatorDefinition,
                           'Pthreads' : pthreadOperatorDefinition}

occaDir = ENV['OCCA_DIR']

hpp = open(occaDir + '/include/occa/operators/virtualDeclarations.hpp', 'w')
hpp.write(virtualOperatorDeclarations(maxN));
hpp.write('\n'); # Make sure there is a newline at the end of the file
hpp.close()

hpp = open(occaDir + '/include/occa/operators/declarations.hpp', 'w')
hpp.write(operatorDeclarations('Base', maxN));
hpp.write('\n');
hpp.close()

hpp = open(occaDir + '/src/operators/definitions.cpp', 'w')
hpp.write(operatorDefinitions('Base', maxN));
hpp.write('\n');
hpp.close()

hpp = open(occaDir + '/src/operators/runFunctionFromArguments.cpp', 'w')
hpp.write(runFunctionFromArguments(maxN));
hpp.write('\n');
hpp.close()

hpp = open(occaDir + '/src/operators/runKernelFromArguments.cpp', 'w')
hpp.write(runKernelFromArguments(maxN));
hpp.write('\n');
hpp.close()

for mode in operatorModeDefinition:
    hpp = open(occaDir + '/include/occa/operators/' + mode + 'KernelOperators.hpp', 'w')
    hpp.write(operatorDeclarations(mode, maxN));
    hpp.write('\n');
    hpp.close()

    cpp = open(occaDir + '/src/operators/' + mode + 'KernelOperators.cpp', 'w')
    cpp.write(operatorDefinitions(mode, maxN));
    cpp.write('\n');
    cpp.close()

hpp = open(occaDir + '/include/occa/operators/cKernelOperators.hpp', 'w')
hpp.write(cOperatorDeclarations(maxN));
hpp.write('\n');
hpp.close()

cpp = open(occaDir + '/src/operators/cKernelOperators.cpp', 'w')
cpp.write(cOperatorDefinitions(maxN));
cpp.write('\n');
cpp.close()
