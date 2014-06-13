classdef type < handle
    properties
        cPtr
    end

    methods
       function this = type(arg, cType)
           this.cPtr = calllib('libocca', occa.typeMatToOCCA(cType), arg);
       end
    end
end