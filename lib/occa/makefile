javaDir = /Library/Java/JavaVirtualMachines/jdk1.8.0_25.jdk/Contents/Home/

all: Device.c
	clang -o Device.so -c -I${OCCA_DIR}/include -L${OCCA_DIR}/lib -I$(javaDir)/include -I$(javaDir)/include/darwin -L$(javaDir)/jre/lib/server Device.c -locca -ljvm