function init
    if libisloaded('libocca')
        return
    end
    
    occaDir = getenv('OCCA_DIR');
    
    occaHeader = strcat(occaDir, '/include/occaCBase.hpp');
    occaLib    = strcat(occaDir, '/lib/libocca.so');
    
    [~, ~] = loadlibrary(occaLib, occaHeader);   
end