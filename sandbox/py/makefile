_C_occa.so: _C_occa.c
	clang -shared -fPIC _C_occa.c -o _C_occa.so -I${OCCA_DIR}/include -L${OCCA_DIR}/lib -I/System/Library/Frameworks/Python.framework//Versions/2.7/include/python2.7/ -I/System/Library/Frameworks/Python.framework//Versions/2.7/Extras/lib/python/numpy/core/include/ -locca -framework Python

clean:
	rm -f _C_occa.so