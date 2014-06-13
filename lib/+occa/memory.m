classdef memory < handle
   properties
       isAllocated = 0
       cMemory
   end
   
   methods       
       function mode_ = mode(this)
           mode_ = calllib('libocca', 'occaMemoryMode', this.cMemory);
       end
       
       function this = memory(cMemory_)
           this.cMemory = cMemory_;
           this.isAllocated = 1;
       end
       
       function copyTo(this, dest, type, varargin)
           argc = length(varargin);
           
           if argc == 2
               entries = varargin{1};
               offset  = varargin{2};
           elseif argc == 1
               entries = varargin{1};
               offset  = 0;
           else
               entries = 0;
               offset  = 0;
           end
           
           if isnumeric(dest) && (entries == 0)
               bytes = numel(dest)*occa.sizeof(type);
           else
               bytes = entries*occa.sizeof(type);
           end
           
           if isnumeric(dest)      
               ptr = libpointer(strcat(type, 'Ptr'), dest);
                              
               calllib('libocca', 'occaCopyMemToPtr', ptr,          ...
                                                      this.cMemory, ...
                                                      bytes,        ...
                                                      offset);
                                                  
               ptr.Value % [-]
           else
               calllib('libocca', 'occaCopyMemToMem', dest.cMemory, ...
                                                      this.cMemory, ...
                                                      bytes,        ...
                                                      offset);
           end
       end
       
       function copyFrom(this, src, type, varargin)
           argc = length(varargin);
           
           if argc == 2
               entries = varargin{1};
               offset  = varargin{2};
           elseif argc == 1
               entries = varargin{1};
               offset  = 0;
           else
               entries = 0;
               offset  = 0;
           end
           
           bytes = entries*occa.sizeof(type);
           
           if isnumeric(src)
               calllib('libocca', 'occaCopyMemFromPtr', this.cMemory, ...
                                                        src,          ...
                                                        bytes,        ...
                                                        offset);
           else
               calllib('libocca', 'occaCopyMemFromMem', this.cMemory, ...
                                                        src.cMemory,  ...
                                                        bytes,        ...
                                                        offset);
           end
       end
       
       function asyncCopyTo(this, dest, type, varargin)
       end
       
       function asyncCopyFrom(this, src, type, varargin)
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