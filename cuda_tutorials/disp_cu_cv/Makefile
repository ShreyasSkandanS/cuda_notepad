CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all:
	nvcc -I. -arch=sm_30 -c stereoDisparity.cu -o stereoDisparity.o
	g++ -o main_file main_file.cpp stereoDisparity.o $(CFLAGS) $(LIBS) -L/usr/local/cuda/lib64 -lcudart

clean: 
	@rm -rf *.o main_file stereoDisparity
	@rm -rf *~
	@rm -rf *.pgm
