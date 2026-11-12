CC = gcc
CFLAGS = -O2 -Wall
LIBS = -lOpenCL

demo: demo.c
	$(CC) $(CFLAGS) -o demo demo.c $(LIBS)

run: demo
	./demo kernels.cl

errores: demo
	./demo errores.cl

ir: kernels.cl
	clang -x cl -cl-std=CL1.2 -Xclang -finclude-default-header -target nvptx64-nvidia-cuda -O2 -S -emit-llvm kernels.cl -o kernels.ll

clean:
	rm -f demo kernels.ptx kernels.ll
