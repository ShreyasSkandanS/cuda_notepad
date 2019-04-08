CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all:
	nvcc -I. -arch=sm_30 -c blur.cu -o blur.o
	g++ -o main_file main_file.cpp blur.o $(CFLAGS) $(LIBS) -L/usr/local/cuda/lib64 -lcudart

clean: 
	@rm -rf *.o main_file blur
	@rm -rf *~
