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
       end
       
       function run(this)
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