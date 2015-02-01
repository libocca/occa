from ctypes import *
import occa

entries = 5

a  = [i     for i in xrange(entries)]
b  = [1 - i for i in xrange(entries)]
ab = [0     for i in xrange(entries)]

Serial_Info   = "mode = Serial"
OpenMP_Info   = 'mode = OpenMP  , schedule = compact, chunk = 10'
OpenCL_Info   = "mode = OpenCL  , platformID = 0, deviceID = 0"
CUDA_Info     = "mode = CUDA    , deviceID = 0"
Pthreads_Info = "mode = Pthreads, threadCount = 4, schedule = compact, pinnedCores = [0, 0, 1, 1]"
COI_Info      = "mode = COI     , deviceID = 0"

device = occa.device(Serial_Info)

o_a  = device.malloc(a , c_float)
o_b  = device.malloc(b , c_float)
o_ab = device.malloc(ab, c_float)

addVectors = device.buildKernelFromSource("addVectors.okl",
                                          "addVectors")

addVectors([c_int(entries),
            o_a, o_b, o_ab])

o_ab.copyTo(ab, c_float)

print ab
