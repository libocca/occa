import occa
import numpy as np

entries = np.int32(10)

addVectors = occa.buildKernel('addVectors.okl', 'addVectors')

a  = occa.managedAlloc(np.float32, entries)
b  = occa.managedAlloc(np.float32, entries)
ab = occa.managedAlloc(np.float32, entries)

for i in xrange(entries):
    a[i]  = i
    b[i]  = 1 - i
    ab[i] = 0

addVectors(entries, a, b, ab)

occa.finish()

print ab