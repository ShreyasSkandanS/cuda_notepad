all: program

program: cudacode.o
	g++ -c file2.cpp
	g++ -o program file1.o file2.o -L/usr/local/cuda/lib64 -lcudart
    
cudacode.o:
	nvcc -c file1.cu

clean:
	rm -rf *o program

