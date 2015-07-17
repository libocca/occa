function occaType = typeMatToOCCA(matType)
    switch matType
    case 'int8'
      occaType = 'occaChar';
    case 'uint8'
      occaType = 'occaUChar';
    case 'int16'
      occaType = 'occaShort';
    case 'uint16'
      occaType = 'occaUShort';
    case 'int32'
      occaType = 'occaInt';
    case 'uint32'
      occaType = 'occaUInt';
    case 'int64'
      occaType = 'occaLong';
    case 'uint64'
      occaType = 'occaULong';

    case 'single'
      occaType = 'occaFloat';
    case 'double'
      occaType = 'occaDouble';
    end
end