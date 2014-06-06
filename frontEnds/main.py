import occa

entries = 5

a  = [i     for i in xrange(entries)]
b  = [1 - i for i in xrange(entries)]
ab = [0     for i in xrange(entries)]

device = occa.device("OpenMP", 0, 0)

o_a  = device.malloc(a);
o_b  = device.malloc(b);
o_ab = device.malloc(ab);

addVectors = device.buildKernelFromSource("addVectors.occa",
                                          "addVectors");

dims = 1;
itemsPerGroup([2]);
groups([(entries + itemsPerGroup - 1)/itemsPerGroup]);

addVectors.setWorkingDims(dims,
                          itemsPerGroup,
                          groups);

o_a.copyFrom(a);
o_b.copyFrom(b);

addVectors([entries, o_a, o_b, o_ab]);

o_ab.copyTo(ab);

print ab

addVectors.free();
o_a.free();
o_b.free();
o_ab.free();
device.free();
