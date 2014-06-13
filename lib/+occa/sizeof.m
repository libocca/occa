function bytes = sizeof(type)    
    eval(strcat('occaTmp = ', type, '(0);'));
    occaTmpInfo = whos('occaTmp');
        
    bytes = occaTmpInfo.bytes;
end