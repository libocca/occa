classdef kernel < handle
   properties
       isAllocated = 0
       cKernel
   end

   methods
       function mode_ = mode(this)
           mode_ = calllib('libocca', 'occaKernelMode', this.cKernel);
       end

       function this = kernel(cKernel_)
           this.cKernel = cKernel_;
           this.isAllocated = 1;
       end

       function size_ = preferredDimSize(this)
           size_ = calllib('libocca', 'occaKernelPreferredDimSize', this.cKernel);
       end

       function setWorkingDims(this, dims, itemsPerGroup, groups)
           if isnumeric(itemsPerGroup)
               ipg = itemsPerGroup(:);
               ipg = [ipg(1:dims); ones(3 - dims, 1)];
           else
               ipg = [itemsPerGroup; ones(2,1)];
           end

           if isnumeric(groups)
               grp = groups(:);
               grp = [grp(1:dims); ones(3 - dims, 1)];
           else
               grp = [groups; ones(2,1)];
           end

           calllib('libocca', 'occaKernelSetAllWorkingDims', this.cKernel, ...
                                                             dims,         ...
                                                             ipg(1),       ...
                                                             ipg(2),       ...
                                                             ipg(3),       ...
                                                             grp(1),       ...
                                                             grp(2),       ...
                                                             grp(3));
       end

       function varargout = subsref(this, index)
           indexArgs = numel(index);

           if indexArgs == 1
               switch index.type
               case '.'
                   switch index.subs
                   case 'isAllocated'
                       varargout{1} = this.isAllocated;
                   case 'cKernel'
                       varargout{1} = this.cKernel;
                   end
               case '()'
                   argList = calllib('libocca', 'occaCreateArgumentList');

                   for pos = 1:length(index.subs)
                       arg = index.subs{pos};

                       if isa(arg, 'occa.memory')
                           calllib('libocca', 'occaArgumentListAddArg', argList, ...
                                                                        pos - 1, ...
                                                                        arg.cMemory);
                       else
                           calllib('libocca', 'occaArgumentListAddArg', argList, ...
                                                                        pos - 1, ...
                                                                        arg.cPtr);
                       end
                   end

                   calllib('libocca', 'occaKernelRun_', this.cKernel, ...
                                                        argList);

                   % calllib('libocca', 'occaArgumentListFree', argList);
               end
           else
               numargout = nargout(index(1).subs);
               [varargout{1:numargout}] = builtin('subsref', this, index);
           end
       end

       function timeTaken_ = timeTaken(this)
           timeTaken_ = calllib('libocca', 'occaKernelTimeTaken', this.cKernel);
       end

       function free(this)
           if this.isAllocated
               timeTaken_ = calllib('libocca', 'occaKernelFree', this.cKernel);
               this.isAllocated = 0;
           end
       end

       function delete(this)
           this.free()
       end
   end
end