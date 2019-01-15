all: ps_hybrid ps_gpu ps_cpu

ps_hybrid: main_hybrid.cu load_data.cpp
	nvcc -Xcompiler -fopenmp -lgomp main_hybrid.cu load_data.cpp -arch=sm_35 \
	     -default-stream per-thread -o ps_hybrid

ps_gpu: main_gpu_only.cu load_data.cpp
	nvcc -Xcompiler -fopenmp -lgomp main_gpu_only.cu load_data.cpp -arch=sm_35 \
	     -default-stream per-thread -o ps_gpu

ps_cpu: main_cpu_only.cpp load_data.cpp
	g++  -fopenmp -lgomp main_cpu_only.cpp load_data.cpp -o ps_cpu

clean:
	rm -f *.o
	rm -f *~
	rm -f ps_hybrid
	rm -f ps_gpu
	rm -f ps_cpu
