classdef memory < handle
   properties
       isAllocated = 0
       cMemory
   end
   
   methods       
       function mode_ = mode(this)
           mode_ = calllib('libocca', 'occaMemoryMode', this.cMemory);
       end
       
       function copyTo(this, dest)
       end
       
       function copyFrom(this, src)
       end
       
       function asyncCopyTo(this, dest)
       end
       
       function asyncCopyFrom(this, src)
       end
       
       function swap(this, m)
           calllib('libocca', 'occaMemorySwap', this.cMemory, m.cMemory);
       end
       
       function free(this)
           if isAllocated
               calllib('libocca', 'occaMemoryFree', this.cMemory);
           end
       end
       
       function delete(this)
           this.free()
       end
   end
end