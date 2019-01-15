# Hybrid Point Solve
A PDE Solver originally implemented in Fortran for the Fun3D (Fully Unstructured 3D Grid) NASA fluid flow simulation ecosystem.  This version has been converted to C and the OpenMP and CUDA language extensions have been utilized to allow the solve routines to exploit parallelism on both the CPU and GPU simultaneously.

## Files
There are three primary files of interest: 1) main_cpu_only.cpp, 2) main_gpu_only.cu, and 3) main_hybrid.cu.  Each of these files follow the same general procedure for loading in the four input file fragments and then either assigning the CPU or GPU to perform that portion of the computation.  For the cpu only version, all four fragments are assigned to the CPU.  For the gpu only version, all four fragments are assigned to the GPU.  For the hybrid version, one fragment is assigned to the CPU and the remaining three are assigned to the GPU.  The results of this division of labor are described in the final report (pdf available in this repository).

The data files are very large (over 500 mb) each and separate arrangements must be made to deliver those if they are required.
