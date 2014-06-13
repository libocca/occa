classdef kernel < handle
   properties
       isAllocated = 0
       cKernel
   end

   methods
       function mode_ = mode(this)
           mode_ = calllib('libocca', 'occaKernelMode', this.cKernel);
       end

       function size_ = preferredDimSize(this)
           size_ = calllib('libocca', 'occaKernelPreferredDimSize', this.cKernel);
       end

       function setWorkingDims(this, dims, itemsPerGroup, groups)
           if isnumeric(itemsPerGroup)
               ipg = itemsPerGroup(:);
               ipg = [ipg(1:(dims + 1)); ones(3 - dims, 1)];
           else
               ipg = [itemsPerGroup; ones(2,1)];
           end

           if isnumeric(groups)
               grp = groups(:);
               grp = [grp(1:(dims + 1)); ones(3 - dims, 1)];
           else
               grp = [groups; ones(2,1)];
           end

           ipgDim = calllib('libocca', 'occaDim', ipg(1), ipg(2), ipg(3));
           grpDim = calllib('libocca', 'occaDim', grp(1), grp(2), grp(3));

           calllib('libocca', 'occaKernelSetWorkingDims', this.cKernel,
                                                          dims,
                                                          ipgDim,
                                                          grpDim);
       end

       function varargout = subsref(this, index)
           switch index.type
           case '.'
               switch index.subs
               case 'isAllocated'
                   varargout{1} = this.isAllocated;
               case 'cKernel'
                   varargout{1} = this.cKernel;
               end
           case '()'
               argList = calllib('libocca', 'occaGenArgumentList');

               pos = 0;
               for arg = index.subs
                   if isa(arg, 'occa.memory')
                       calllib('libocca', 'occaArgumentListAddArg', argList, ...
                                                                    pos,     ...
                                                                    arg.cMemory);
                   else
                       calllib('libocca', 'occaArgumentListAddArg', argList, ...
                                                                    pos,     ...
                                                                    arg.cPtr);
                   end

                   pos = pos + 1;
               end
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