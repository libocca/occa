using OCCA

entries = convert(Int32, 10)

addVectors = OCCA.build_kernel("addVectors.okl",
                               "addVectors")

a  = OCCA.malloc(Float32, entries, managed = true)
b  = OCCA.malloc(Float32, entries, managed = true)
ab = OCCA.malloc(Float32, entries, managed = true)

for i in 1:entries
    a[i]  = (1 - i)
    b[i]  = i
    ab[i] = 0
end

OCCA.call(addVectors,
          entries, a, b, ab)

OCCA.finish()

println(ab)