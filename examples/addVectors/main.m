entries = 5;

a  = ones(entries, 1);
b  = ones(entries, 1);
ab = zeros(entries, 1);

Serial_Info   = 'mode = Serial';
OpenMP_Info   = 'mode = OpenMP  , schedule = compact, chunk = 10';
OpenCL_Info   = 'mode = OpenCL  , platformID = 0, deviceID = 0';
CUDA_Info     = 'mode = CUDA    , deviceID = 0';
Pthreads_Info = 'mode = Pthreads, threadCount = 4, schedule = compact, pinnedCores = [0, 0, 1, 1]';
COI_Info      = 'mode = COI     , deviceID = 0';

device = occa.device(Serial_Info);

o_a  = device.malloc(a , 'single');
o_b  = device.malloc(b , 'single');
o_ab = device.malloc(ab, 'single');

addVectors = device.buildKernelFromSource('addVectors.okl', ...
                                          'addVectors');

addVectors(occa.type(entries, 'int32'), ...
           o_a, o_b, o_ab);

ab = o_ab(:);

ab
