require( bytestring(ENV["OCCA_DIR"], "/lib/occa.jl") )

entries = 5

Serial_Info   = "mode = Serial"
OpenMP_Info   = "mode = OpenMP  , schedule = compact, chunk = 10";
OpenCL_Info   = "mode = OpenCL  , platformID = 0, deviceID = 0";
CUDA_Info     = "mode = CUDA    , deviceID = 0";
Pthreads_Info = "mode = Pthreads, threadCount = 4, schedule = compact, pinnedCores = [0, 0, 1, 1]";
COI_Info      = "mode = COI     , deviceID = 0";

device = occa.device(Serial_Info);

a  = Float32[1 - i for i in 1:entries]
b  = Float32[i     for i in 1:entries]
ab = Float32[0     for i in 1:entries]

o_a  = occa.malloc(device, a);
o_b  = occa.malloc(device, b);
o_ab = occa.malloc(device, ab);

addVectors = occa.buildKernelFromSource(device,
                                        "addVectors.okl",
                                        "addVectors")

occa.runKernel(addVectors,
               (entries, Int32),
               o_a, o_b, o_ab)

occa.memcpy(ab, o_ab)

println(ab)
