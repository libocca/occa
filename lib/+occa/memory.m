classdef memory < handle
   properties
       isAllocated = 0
       cMemory
       cType
       cPtrType
       cSize
       cEntries
   end

   methods
       function mode_ = mode(this)
           mode_ = calllib('libocca', 'occaMemoryMode', this.cMemory);
       end

       function this = memory(cMemory_)
           this.cMemory = cMemory_;
           this.isAllocated = 1;
       end

       function varargout = subsref(this, index)
           switch index.type
           case '.'
               switch index.subs
               case 'isAllocated'
                   varargout{1} = this.isAllocated;
               case 'cMemory'
                   varargout{1} = this.cMemory;
               case 'cType'
                   varargout{1} = this.cType;
               case 'cPtrType'
                   varargout{1} = this.cPtrType;
               case 'cSize'
                   varargout{1} = this.cSize;
               case 'cEntries'
                   varargout{1} = this.cEntries;
               end
           case '()'
               if isnumeric(index.subs{1})
                   entries = numel(index.subs{1});
                   offset  = (index.subs{1}(1) - 1);
               else
                   switch index.subs{1}
                   case ':'
                       entries = this.cEntries;
                       offset  = 0;
                   otherwise
                       entries = 1;
                       offset  = (index.subs{1} - 1);
                   end
               end

               bytes  = entries*occa.sizeof(this.cType);
               offset = offset*occa.sizeof(this.cType);

               ptr = libpointer(this.cPtrType, zeros(entries,1));

               calllib('libocca', 'occaCopyMemToPtr', ptr,          ...
                                                      this.cMemory, ...
                                                      bytes,        ...
                                                      offset);

               varargout{1} = ptr.value;
           end
       end

       function this = subsasgn(this, index, value)
           switch index.type
           case '.'
               switch index.subs
               case 'isAllocated'
                   this.isAllocated = value;
               case 'cMemory'
                   this.cMemory = value;
               case 'cType'
                   this.cType = value;
               case 'cPtrType'
                   this.cPtrType = value;
               case 'cSize'
                   this.cSize = value;
               case 'cEntries'
                   this.cEntries = value;
               end
           case '()'
               if isnumeric(index.subs{1})
                   entries = numel(index.subs{1});
                   offset  = (index.subs{1}(1) - 1);
               else
                   switch index.subs{1}
                   case ':'
                       entries = this.cEntries;
                       offset  = 0;
                   otherwise
                       entries = 1;
                       offset  = (index.subs{1} - 1);
                   end
               end

               bytes  = entries*occa.sizeof(this.cType);
               offset = offset*occa.sizeof(this.cType);

               ptr = libpointer(this.cPtrType, value);

               calllib('libocca', 'occaCopyPtrToMem', this.cMemory, ...
                                                      ptr,          ...
                                                      bytes,        ...
                                                      offset);
           end
       end

       function copyToMem(this, dest, type, varargin)
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

           calllib('libocca', 'occaCopyMemToMem', dest.cMemory, ...
                                                  this.cMemory, ...
                                                  bytes,        ...
                                                  offset);
       end

       function copyFromMem(this, src, type, varargin)
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

           calllib('libocca', 'occaCopyMemToMem', this.cMemory, ...
                                                  src.cMemory,  ...
                                                  bytes,        ...
                                                  offset);
       end

       function swap(this, m)
           calllib('libocca', 'occaMemorySwap', this.cMemory, m.cMemory);
       end

       function free(this)
           if this.isAllocated
               calllib('libocca', 'occaMemoryFree', this.cMemory);
               this.isAllocated = 0;
           end
       end

       function delete(this)
           this.free()
       end
   end
end