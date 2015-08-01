javaDir = /Library/Java/JavaVirtualMachines/jdk1.8.0_25.jdk/Contents/Home/

all: occa_Device.h
	clang -o occa_Device.so -c -I${OCCA_DIR}/include -L${OCCA_DIR}/lib -I$(javaDir)/include -I$(javaDir)/include/darwin -L$(javaDir)/jre/lib/server occa_Device.h -locca -ljvm