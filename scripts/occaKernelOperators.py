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

def functionPointerTypeDefs(N):
    return '\n\n'.join([functionPointerTypeDef(n + 1) for n in xrange(N)])

def functionPointerTypeDef(N):
    return 'typedef void (*functionPointer' + str(N) + ' )(int *occaKernelInfoArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2, ' +  ' '.join(['void *arg' + str(n) + vnlc(n, N) for n in xrange(N)]) + ');'

def coiFunctionPointerTypeDefs(N):
    return '\n\n'.join([coiFunctionPointerTypeDef(n + 1) for n in xrange(N)])

def coiFunctionPointerTypeDef(N):
    return 'typedef void (*functionPointer' + str(N) + ' )(int *occaKernelInfoArgs, ' +  ' '.join(['void *arg' + str(n) + vnlc(n, N) for n in xrange(N)]) + ');'

def runFromArguments(N):
    return 'switch(argumentCount){\n' + '\n'.join([runFromArgument(n + 1) for n in xrange(N)]) + '}'

def runFromArgument(N):
    return '  case ' + str(N) + """:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(""" + ', '.join(['arguments[{0}]'.format(n) for n in xrange(N)]) + """);
  }""" + (("""
  else{
    (*kHandle)(kHandle->nestedKernels, """ + ', '.join(['arguments[{0}]'.format(n) for n in xrange(N)]) + """);
  }""") if (N < maxN) else '') + """
  break;"""

def virtualOperatorDeclarations(N):
    return '\n\n'.join([virtualOperatorDeclaration(n + 1) for n in xrange(N)])

def virtualOperatorDeclaration(N):
    return '    virtual void operator () ({0}) = 0;'.format( ' '.join(['const kernelArg &arg' + str(n) + vnlc(n, N) for n in xrange(N)]) )

def operatorDeclarations(mode, N):
    return '\n\n'.join([operatorDeclaration(mode, n + 1) for n in xrange(N)])

def operatorDeclaration(mode, N):
    if mode == 'Base':
        ret = '    void operator () ({0});'.format( ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in xrange(N)]) )
    else:
        ret = '    template <>\n'\
              + '    void kernel_t<{0}>::operator () ({1});'.format(mode, ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in xrange(N)]) )

    if mode == 'Pthreads':
        ret += '\n    static void launchKernel{0}(PthreadKernelArg_t &args);'.format(N)

    return ret

def operatorDefinitions(mode, N):
    return '\n\n'.join([operatorDefinition(mode, n + 1) for n in xrange(N)])

def operatorDefinition(mode, N):
    if mode == 'Base':
        return """  void kernel::operator() (""" + ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in xrange(N)]) + """){
    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(""" + ' '.join(['arg' + str(n) + nlc(n, N) for n in xrange(N)]) + """);
    }
    else{""" + (("""
      (*kHandle)(kHandle->nestedKernels, """ + ' '.join(['arg' + str(n) + nlc(n, N) for n in xrange(N)]) + """);""") \
                if (N < maxN) else '') + """
    }
  }

  void kernelDatabase::operator() (""" + ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in xrange(N)]) + """){
    occa::device *launchDevice = NULL;

    if(arg0.dev) launchDevice = arg0.dev;
    """ + '    '.join([('else if(arg' + str(n + 1) + '.dev) launchDevice = arg' + str(n + 1) + '.dev;\n') for n in xrange(N - 1)]) + """
    (*this)[*launchDevice](""" + ' '.join(['arg' + str(n) + nlc(n, N) for n in xrange(N)]) + """);
  }"""
    else:
        header = operatorDefinitionHeader(mode, N)
        return header + operatorModeDefinition[mode](N) + "\n  }"

def operatorDefinitionHeader(mode, N):
    return """  template <>
  void kernel_t<{0}>::operator () ({1}){{""".format(mode, ' '.join(['const kernelArg &arg' + str(n) + nlc(n, N) for n in xrange(N)]))

def pthreadOperatorDefinition(N):
    return """
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      """ + '\n    '.join(['args->args[{0}] = arg{0};'.format(n) for n in xrange(N)]) + """

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel""" + str(N) + """);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel""" + str(N) + """(PthreadKernelArg_t &args){
    functionPointer""" + str(N) + """ tmpKernel = (functionPointer""" + str(N) + """) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12] = {outer.z, outer.y, outer.x,
                              inner.z, inner.y, inner.x,
                              start.z, end.z,
                              start.y, end.y,
                              start.x, end.x};

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              """ + ',\n            '.join(['args.args[{0}].data()'.format(n) for n in xrange(N)]) + """);

    delete &args;"""

def ompOperatorDefinition(N):
    return """
    OpenMPKernelData_t &data_ = *((OpenMPKernelData_t*) data);
    functionPointer""" + str(N) + """ tmpKernel = (functionPointer""" + str(N) + """) data_.handle;
    int occaKernelArgs[6] = {outer.z, outer.y, outer.x,
                             inner.z, inner.y, inner.x};

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              """ + ',\n              '.join(['arg{0}.data()'.format(n) for n in xrange(N)]) + ');'

def clOperatorDefinition(N):
    return """
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    """ + '\n\n    '.join(["""OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg{0}.size, arg{0}.data()));
    if(arg{0}.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg{0}.arg2.void_));""".format(n) for n in xrange(N)]) + """

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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
    int argCount = 0;

    void *args[""" + str(2 * N) + """];

    args[argCount++] = &occaKernelInfoArgs;

    """ + '\n    '.join([('args[argCount++] = arg{0}.pointer ? (arg{0}.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg{0}.arg.void_)->surface) : arg{0}.arg.void_) : (void*) &arg{0}.arg;\n    ' + \
                          'if(arg{0}.hasTwoArgs)\n    '             + \
                          '  args[argCount++] = arg{0}.arg2.void_;').format(n) for n in xrange(N)]) + """

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);"""

def coiOperatorDefinition(N):
    return """
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dev->dHandle)->data);

    int occaKernelArgs[6] = {outer.z, outer.y, outer.x,
                             inner.z, inner.y, inner.x};

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
    }}""".format(n) for n in xrange(N)]) + """

    coiStream &stream = *((coiStream*) dev->currentStream);

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

def cOperatorDeclarations(N):
    return '\n\n'.join([cOperatorDeclaration(n + 1) for n in xrange(N)])

def cOperatorDeclaration(N):
    return '    OCCA_LFUNC void OCCA_RFUNC occaKernelRun{0}(occaKernel kernel, {1});\n'.format(N, ' '.join(['void *arg' + str(n) + nlc(n, N) for n in xrange(N)]) )

def cOperatorDefinitions(N):
    return '\n\n'.join([cOperatorDefinition(n + 1) for n in xrange(N)])

def cOperatorDefinition(N):
    addArguments = ''

    for n in xrange(N):
        addArguments += ('    {\n' + \
                         '      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg' + str(n) + ');\n' + \
                         '      if(__occa_memory__.type == 0){\n' + \
                         '        __occa_kernel__.addArgument(' + str(n) + ', occa::kernelArg(__occa_memory__.mem));\n' + \
                         '      }\n' + \
                         '      else{\n' + \
                         '        occaType_t &__occa_type__ = *((occaType_t*) arg' + str(n) + ');\n' + \
                         '        __occa_kernel__.addArgument(' + str(n) + ', occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));\n' + \
                         '      }\n' + \
                         '    }\n')

    return ('    void OCCA_RFUNC occaKernelRun{0}(occaKernel kernel, {1}){{\n'.format(N, ' '.join(['void *arg' + str(n) + nlc(n, N) for n in xrange(N)]) ) + \
            '      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);\n' + \
            '    __occa_kernel__.clearArgumentList();\n' + \
            addArguments + \
            '    __occa_kernel__.runFromArguments();\n' + \
            '    }\n');

operatorModeDefinition = { 'Pthreads' : pthreadOperatorDefinition,
                           'OpenMP'   : ompOperatorDefinition,
                           'OpenCL'   : clOperatorDefinition,
                           'CUDA'     : cudaOperatorDefinition,
                           'COI'      : coiOperatorDefinition }

hpp = open('../include/operators/occaVirtualOperatorDeclarations.hpp', 'w')
hpp.write(virtualOperatorDeclarations(maxN));
hpp.close()

hpp = open('../include/operators/occaOperatorDeclarations.hpp', 'w')
hpp.write(operatorDeclarations('Base', maxN));
hpp.close()

hpp = open('../include/operators/occaFunctionPointerTypeDefs.hpp', 'w')
hpp.write(functionPointerTypeDefs(maxN));
hpp.close()

hpp = open('../include/operators/occaCOIFunctionPointerTypeDefs.hpp', 'w')
hpp.write(coiFunctionPointerTypeDefs(maxN));
hpp.close()

hpp = open('../src/operators/occaOperatorDefinitions.cpp', 'w')
hpp.write(operatorDefinitions('Base', maxN));
hpp.close()

hpp = open('../src/operators/occaRunFromArguments.cpp', 'w')
hpp.write(runFromArguments(maxN));
hpp.close()

for mode in operatorModeDefinition:
    hpp = open('../include/operators/occa' + mode + 'KernelOperators.hpp', 'w')
    hpp.write(operatorDeclarations(mode, maxN));
    hpp.close()

    cpp = open('../src/operators/occa' + mode + 'KernelOperators.cpp', 'w')
    cpp.write(operatorDefinitions(mode, maxN));
    cpp.close()

hpp = open('../include/operators/occaCKernelOperators.hpp', 'w')
hpp.write(cOperatorDeclarations(maxN));
hpp.close()

cpp = open('../src/operators/occaCKernelOperators.cpp', 'w')
cpp.write(cOperatorDefinitions(maxN));
cpp.close()
