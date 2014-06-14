include("occa.jl")

entries = 5

device = occa.device("OpenMP", 0, 0);

# Dynamic range?
a  = [convert(Float32, 1 - i) for i in 1:5]
b  = [convert(Float32, i    ) for i in 1:5]
ab = [convert(Float32, 0    ) for i in 1:5]

o_a  = occa.malloc(device, a);
o_b  = occa.malloc(device, b);
o_ab = occa.malloc(device, ab);

addVectors = occa.buildKernelFromSource(device,
                               "addVectors.occa",
                               "addVectors")

dims = 1;
itemsPerGroup = 2;
groups = (entries + itemsPerGroup - 1)/itemsPerGroup;

occa.setWorkingDims(addVectors,
                    dims, itemsPerGroup, groups);

occa.runKernel(addVectors,
               (entries, Int32),
               o_a, o_b, o_ab)

occa.memcpy(ab, o_ab)

println(ab)
