classdef device < handle
    properties
        isAllocated = 0
        cDevice
    end

    methods (Static)
        function deviceCountFunction(arg)
            persistent counter;

            if arg == 1
                if isempty(counter)
                    counter = 1;
                else
                    counter = counter + 1;
                end
            elseif arg == 0
                counter = counter - 1;
            end

            if counter == 0
                unloadlibrary('libocca');
            end
        end
    end

    methods
        function mode_ = mode(this)
            mode_ = calllib('libocca', 'occaDeviceMode', this.cDevice);
        end

        function this = device(infos)
            occa.init()

            this.deviceCountFunction(1)

            this.cDevice     = calllib('libocca', 'occaGetDevice', infos);
            this.isAllocated = 1;
        end

        function setup(this, infos)
            occa.init()

            this.deviceCountFunction(1)

            this.cDevice     = calllib('libocca', 'occaGetDevice', infos);
            this.isAllocated = 1;
        end

        function setCompiler(this, compiler)
            calllib('libocca', 'occaDeviceSetCompiler', this.cDevice, compiler);
        end

        function setCompilerFlags(this, compilerFlags)
            calllib('libocca', 'occaDeviceSetCompilerFlags', this.cDevice, compilerFlags);
        end

        function kernel_ = buildKernelFromSource(this, filename, functionName, varargin)
            if nargin == 3
                info = libpointer
            else
                info = varargin{1}
            end

            cKernel = calllib('libocca', 'occaBuildKernelFromSource', this.cDevice, ...
                                                                      filename,     ...
                                                                      functionName, ...
                                                                      info);

            kernel_ = occa.kernel(cKernel);
            kernel_.isAllocated = 1;
        end

        function kernel_ = buildKernelFromBinary(this, filename, functionName)
            cKernel = calllib('libocca', 'occaBuildKernelFromBinary', this.cDevice, ...
                                                                      filename,     ...
                                                                      functionName);

            kernel_ = occa.kernel(cKernel);
            kernel_.isAllocated = 1;
        end

        function memory_ = malloc(this, arg, type)
            ptrType = strcat(type, 'Ptr');

            if isnumeric(arg)
                bytes = numel(arg)*occa.sizeof(type);

                cMemory = calllib('libocca', 'occaDeviceMalloc', this.cDevice, ...
                                                                 bytes,        ...
                                                                 libpointer(ptrType, arg));
            else
                bytes = arg*occa.sizeof(type);

                cMemory = calllib('libocca', 'occaDeviceMalloc', this.cDevice, bytes, libpointer);
            end

            memory_ = occa.memory(cMemory);

            memory_.cType    = type;
            memory_.cPtrType = ptrType;

            if isnumeric(arg)
                memory_.cSize    = size(arg);
                memory_.cEntries = numel(arg);
            else
                memory_.cSize    = [arg 1];
                memory_.cEntries = arg;
            end
        end

        function stream_ = createStream(this)
            stream_.cStream = calllib('libocca', 'occaCreateStream', this.cDevice);
        end

        function stream_ = getStream(this)
            stream_.cStream = calllib('libocca', 'occaGetStream', this.cDevice);
        end

        function setStream(this, stream_)
            calllib('libocca', 'occaSetStream', this.cDevice, stream_.cStream);
        end

        function free(this)
            if this.isAllocated
                calllib('libocca', 'occaDeviceFree', this.cDevice);

                this.isAllocated = 0;
                this.deviceCountFunction(0)
            end
        end

        function delete(this)
            this.free()
        end
    end
end