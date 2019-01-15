#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <time.h>
//#include <conio.h>
#include <math.h>

#include <omp.h>
#include <cuda_runtime.h>

#include "device_launch_parameters.h"
#include "load_data.h"

#define A_OFF(i,j,k) d_a_off[(i)+((j)*nb)+((k)*nb*nb)]
#define A_DIAG_LU(i,j,k) d_a_diag_lu[(i)+((j)*nb)+((k)*nb*nb)]
#define GPU_DQ(i,j) gpuDq[(i)+((j)*nb)]
#define RES(i, j) d_res[(i)+((j)*nb)]

typedef struct StartEndPair {
  int32_t start, end;
} StartEndPair;

size_t max_val(size_t m, size_t n) {
  return m > n ? m : n;
}

void localMalloc(float** dq, size_t dataSize);
void unifiedMalloc(float** dq, size_t dataSize);
__global__ void solveOnGPU(float *gpuDq, double *d_res, int32_t *d_iam, int32_t *d_jam,
			   float *d_a_off, double *d_a_diag_lu, size_t start, 
			   size_t end, int32_t solve_backwards, int32_t nb);
void solveOnCPU(float *cpuDq, FeedData *cpuData, size_t start, size_t end);
void startDataTransfer(float* dq, size_t row, size_t col, SendRecvType* sr_loc);
void completeDataTransfer(float* dq, size_t row, size_t col, SendRecvType* sr_loc, 
			  float* buffer[], int32_t* sendproc[], size_t const &rank);
void load_data_on_gpu(FeedData *gpuData, int32_t **d_iam, int32_t **d_jam, 
		      double **d_res, double **d_a_diag_lu, float ** d_a_off);
StartEndPair* determineStartEnd(FeedData *data, size_t const &color, size_t const &ipass);

double TimeSpecToSeconds(struct timespec* ts) {
  return (double)ts->tv_sec + (double)ts->tv_nsec/1000000000.0;
}

int main() {

  int32_t i, j;
  int32_t sweep, ipass;
  StartEndPair *cpu_pair = NULL, *gpu1_pair = NULL, *gpu2_pair = NULL, *gpu3_pair = NULL;
  int32_t color, loop_colored_sweeps;
  int32_t sweep_start, sweep_end, sweep_stride;
  FeedData *cpuData, *gpuData_1, *gpuData_2, *gpuData_3;
  cpuData = (FeedData *)malloc(sizeof(FeedData));
  gpuData_1 = (FeedData *)malloc(sizeof(FeedData));
  gpuData_2 = (FeedData *)malloc(sizeof(FeedData));
  gpuData_3 = (FeedData *)malloc(sizeof(FeedData));
  float *cpuDq, *gpuDq_1, *gpuDq_2, *gpuDq_3;

  int32_t *d_iam_1 = NULL, *d_jam_1 = NULL;
  double* d_res_1 = NULL; //2d Array
  double* d_a_diag_lu_1 = NULL; //3dArray
  float* d_a_off_1 = NULL; //3dArray

  int32_t *d_iam_2 = NULL, *d_jam_2 = NULL;
  double* d_res_2 = NULL; //2d Array
  double* d_a_diag_lu_2 = NULL; //3dArray
  float* d_a_off_2 = NULL; //3dArray

  int32_t *d_iam_3 = NULL, *d_jam_3 = NULL;
  double* d_res_3 = NULL; //2d Array
  double* d_a_diag_lu_3 = NULL; //3dArray
  float* d_a_off_3 = NULL; //3dArray
  float* buffer[4];
  int32_t* sendproc[4];

  double wtime1, wtime2, totalTransferTime = 0.0f;

  printf("Loading Data ... \n");
#pragma omp parallel
  {
#pragma omp single nowait
    {
      cpuDq = load_data("4-linear_system.dat.0", cpuData, localMalloc);
    }
#pragma omp single nowait
    {
      gpuDq_1 = load_data("4-linear_system.dat.1", gpuData_1, unifiedMalloc);
    }
#pragma omp single nowait
    {
      gpuDq_2 = load_data("4-linear_system.dat.2", gpuData_2, unifiedMalloc);
    }
#pragma omp single nowait
    {
      gpuDq_3 = load_data("4-linear_system.dat.3", gpuData_3, unifiedMalloc);
    }
  }
#pragma omp barrier

  printf("Data Loaded successfully... \n");
  printf("Loading data to GPU...\n");
  load_data_on_gpu(gpuData_1, &d_iam_1, &d_jam_1, &d_res_1, &d_a_diag_lu_1, &d_a_off_1);
  load_data_on_gpu(gpuData_2, &d_iam_2, &d_jam_2, &d_res_2, &d_a_diag_lu_2, &d_a_off_2);
  load_data_on_gpu(gpuData_3, &d_iam_3, &d_jam_3, &d_res_3, &d_a_diag_lu_3, &d_a_off_3);
  printf("Data loaded to GPU successfully! \n");

  printf("Solving Ax=b...\n");
  int32_t outer_sweeps = cpuData->outer_sweeps;
  struct timespec begin;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  dim3 blockSize(32, 4, 1);
  cudaStream_t stream0;
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  for (i = 0; i < outer_sweeps; ++i)
    {

      loop_colored_sweeps = cpuData->max_colored_sweeps;

      sweep_start = 0;
      sweep_end = loop_colored_sweeps;
      sweep_stride = 1;

      if (cpuData->solve_backwards > 1 || cpuData->solve_backwards < -1) {
	sweep_start = loop_colored_sweeps - 1;
	sweep_end = -1;
	sweep_stride = -1;
      }

      if (cpuData->n_eqns <= 0) {
	sweep_start = 0;
	sweep_end = 1;
	sweep_stride = -1;
      }

      for (sweep = 0; sweep < cpuData->n_sweeps; ++sweep) { // sweeping
	for (color = sweep_start; color < sweep_end; color += sweep_stride) { // color_sweeps
	  for (ipass = 1; ipass <= 2; ++ipass) { // ipass

            #pragma omp parallel
	    {
              #pragma omp single nowait
	      {
		gpu1_pair = determineStartEnd(gpuData_1, color, ipass);
		dim3 numBlock_1((gpu1_pair->end - gpu1_pair->start + 1) / 4 + 1, 1, 1);
		// calling kernel for gpu computation
		solveOnGPU << <numBlock_1, blockSize, 0, stream0 >> > 
		  (gpuDq_1, d_res_1, d_iam_1, d_jam_1, d_a_off_1, d_a_diag_lu_1,
		   gpu1_pair->start, gpu1_pair->end, gpuData_1->solve_backwards, gpuData_1->nb);
	      }
              #pragma omp single nowait
	      {
		gpu2_pair = determineStartEnd(gpuData_2, color, ipass);
		dim3 numBlock_2((gpu2_pair->end - gpu2_pair->start + 1) / 4 + 1, 1, 1);
		// calling kernel for gpu computation
		solveOnGPU << <numBlock_2, blockSize, 0, stream1 >> > 
		  (gpuDq_2, d_res_2, d_iam_2, d_jam_2, d_a_off_2, d_a_diag_lu_2,
	           gpu2_pair->start, gpu2_pair->end, gpuData_2->solve_backwards, gpuData_2->nb);
	      }
              #pragma omp single nowait
	      {
		gpu3_pair = determineStartEnd(gpuData_3, color, ipass);
		dim3 numBlock_3((gpu3_pair->end - gpu3_pair->start + 1) / 4 + 1, 1, 1);
		// calling kernel for pgu computation
		solveOnGPU << <numBlock_3, blockSize, 0, stream2 >> > 
		  (gpuDq_3, d_res_3, d_iam_3, d_jam_3, d_a_off_3, d_a_diag_lu_3,
		   gpu3_pair->start, gpu3_pair->end, gpuData_3->solve_backwards, gpuData_3->nb);
	      }
	    }
					
	    //// GPU computation starts here
	    //gpu1_pair = determineStartEnd(gpuData_1, color, ipass);
	    //dim3 numBlock_1((gpu1_pair->end - gpu1_pair->start + 1) / 4 + 1, 1, 1);
	    //// calling kernel for gpu computation 
	    //solveOnGPU << <numBlock_1, blockSize >> > 
	    //(gpuDq_1, d_res_1, d_iam_1, d_jam_1, d_a_off_1, d_a_diag_lu_1,
	    //	gpu1_pair->start, gpu1_pair->end, gpuData_1->solve_backwards, gpuData_1->nb);

	    //gpu2_pair = determineStartEnd(gpuData_2, color, ipass);
	    //dim3 numBlock_2((gpu2_pair->end - gpu2_pair->start + 1) / 4 + 1, 1, 1);
	    //// calling kernel for gpu computation 
	    //solveOnGPU << <numBlock_2, blockSize >> > 
	    //(gpuDq_2, d_res_2, d_iam_2, d_jam_2, d_a_off_2, d_a_diag_lu_2,
	    //	gpu2_pair->start, gpu2_pair->end, gpuData_2->solve_backwards, gpuData_2->nb);

	    //gpu3_pair = determineStartEnd(gpuData_3, color, ipass);
	    //dim3 numBlock_3((gpu3_pair->end - gpu3_pair->start + 1) / 4 + 1, 1, 1);
	    //// calling kernel for gpu computation 
	    //solveOnGPU << <numBlock_3, blockSize >> > 
	    //(gpuDq_3, d_res_3, d_iam_3, d_jam_3, d_a_off_3, d_a_diag_lu_3,
	    //	gpu3_pair->start, gpu3_pair->end, gpuData_3->solve_backwards, gpuData_3->nb);

	    // cpu computation starts here
	    cpu_pair = determineStartEnd(cpuData, color, ipass);
	    solveOnCPU(cpuDq, cpuData, cpu_pair->start, cpu_pair->end);
	    // Wait for GPU to finish before accessing on host
	    cudaDeviceSynchronize();

            #pragma omp barrier
	    switch (ipass) {
	    case 1:
	      {
		wtime1 = omp_get_wtime();
		/*startDataTransfer(cpuDq, cpuData->dq_dim, cpuData->neqmax, &cpuData->sr[color]);
		  startDataTransfer(gpuDq_1, gpuData_1->dq_dim, gpuData_1->neqmax, 
                                    &gpuData_1->sr[color]);
		  startDataTransfer(gpuDq_2, gpuData_2->dq_dim, gpuData_2->neqmax, 
		                    &gpuData_2->sr[color]);
		  startDataTransfer(gpuDq_3, gpuData_3->dq_dim, gpuData_3->neqmax, 
		                    &gpuData_3->sr[color]);*/
                #pragma omp parallel
		{
                  #pragma omp single nowait
		  {
		    startDataTransfer(cpuDq, cpuData->dq_dim, cpuData->neqmax, &cpuData->sr[color]);
		  }

                  #pragma omp single nowait
		  {
		    startDataTransfer(gpuDq_1, gpuData_1->dq_dim, gpuData_1->neqmax, 
				      &gpuData_1->sr[color]);
		  }

                  #pragma omp single nowait
		  {
		    startDataTransfer(gpuDq_2, gpuData_2->dq_dim, gpuData_2->neqmax, 
		    &gpuData_2->sr[color]);
		  }

                  #pragma omp single nowait
		  {
		    startDataTransfer(gpuDq_3, gpuData_3->dq_dim, gpuData_3->neqmax, 
				      &gpuData_3->sr[color]);
		  }
		}
               #pragma omp barrier
		wtime1 = omp_get_wtime() - wtime1;
		break;
	}
	case 2:
	  {
	  buffer[0] = cpuData->sr[color].single_mat_senddata;
	  buffer[1] = gpuData_1->sr[color].single_mat_senddata;
	  buffer[2] = gpuData_2->sr[color].single_mat_senddata;
	  buffer[3] = gpuData_3->sr[color].single_mat_senddata;
	  sendproc[0] = cpuData->sr[color].sendproc;
	  sendproc[1] = gpuData_1->sr[color].sendproc;
	  sendproc[2] = gpuData_2->sr[color].sendproc;
	  sendproc[3] = gpuData_3->sr[color].sendproc;
	  wtime2 = omp_get_wtime();
	  /*completeDataTransfer(cpuDq, cpuData->dq_dim, cpuData->neqmax, &cpuData->sr[color], 
	    buffer, sendproc, 0);
	    completeDataTransfer(gpuDq_1, gpuData_1->dq_dim, gpuData_1->neqmax, 
	    &gpuData_1->sr[color], buffer, sendproc, 1);
	    completeDataTransfer(gpuDq_2, gpuData_2->dq_dim, gpuData_2->neqmax, 
	    &gpuData_2->sr[color], buffer, sendproc, 2);
	    completeDataTransfer(gpuDq_3, gpuData_3->dq_dim, gpuData_3->neqmax, 
	    &gpuData_3->sr[color], buffer, sendproc, 3);*/
                #pragma omp parallel //shared(buffer, sendproc)
		{
                  #pragma omp single nowait
		  {
		    completeDataTransfer(cpuDq, cpuData->dq_dim, cpuData->neqmax, 
		                         &cpuData->sr[color], buffer, sendproc, 0);
		  }

                  #pragma omp single nowait
		  {
		    completeDataTransfer(gpuDq_1, gpuData_1->dq_dim, gpuData_1->neqmax, 
	                                 &gpuData_1->sr[color], buffer, sendproc, 1);
		  }

                  #pragma omp single nowait
 		  {
		    completeDataTransfer(gpuDq_2, gpuData_2->dq_dim, gpuData_2->neqmax, 
		                         &gpuData_2->sr[color], buffer, sendproc, 2);
		  }

                  #pragma omp single nowait
		  {
	            completeDataTransfer(gpuDq_3, gpuData_3->dq_dim, gpuData_3->neqmax, 
	                                 &gpuData_3->sr[color], buffer, sendproc, 3);
		  }
		}
                #pragma omp barrier
		wtime2 = omp_get_wtime() - wtime2;
		break;
	      }
	    }

	  } // end pass loop
	  totalTransferTime += (wtime1 + wtime2);
	} // end color_sweeps loop
      } // end sweeping loop
    }
  struct timespec finish;
  clock_gettime(CLOCK_MONOTONIC, &finish);
  double time_spent = TimeSpecToSeconds(&finish) - TimeSpecToSeconds(&begin);

  printf("Elapsed: %f seconds\n", time_spent);
  printf("Total transfer time: %f \n", totalTransferTime);
	
  double tol = 1.e-07;
  double  rms1, rms2, rms3, rms4, rms_sum;

  int32_t neq;
  rms1 = 0.0;
  for (i = 0; i < cpuData->neq0; i++) {
    for (j = 0; j < cpuData->dq_dim; j++) {
      rms1 = rms1 + pow(*(cpuDq + i * cpuData->dq_dim + j), 2);
    }
  }

  rms2 = 0.0;
  for (i = 0; i < gpuData_1->neq0; i++) {
    for (j = 0; j < gpuData_1->dq_dim; j++) {
      rms2 = rms2 + pow(*(gpuDq_1 + i * gpuData_1->dq_dim + j), 2);
    }
  }

  rms3 = 0.0;
  for (i = 0; i < gpuData_2->neq0; i++) {
    for (j = 0; j < gpuData_2->dq_dim; j++) {
      rms3 = rms3 + pow(*(gpuDq_2 + i * gpuData_2->dq_dim + j), 2);
    }
  }

  rms4 = 0.0;
  for (i = 0; i < gpuData_3->neq0; i++) {
    for (j = 0; j < gpuData_3->dq_dim; j++) {
      rms4 = rms4 + pow(*(gpuDq_3 + i * gpuData_3->dq_dim + j), 2);
    }
  }

  rms_sum = rms1 + rms2 + rms3 + rms4;
  neq = cpuData->neq0 + gpuData_1->neq0 + gpuData_2->neq0 + gpuData_3->neq0;
  rms_sum = sqrt(rms_sum / (double)neq / (double)cpuData->nb);

  if (isnan(rms_sum) || (rms_sum - cpuData->golden) >= tol) {
    printf("test failed: %.10E", rms_sum - cpuData->golden);
  }
  else {
    printf("test passed, result = %.10E\tgolden = %.10E\n", rms_sum, cpuData->golden);
  }
	
  /*char nop[20];
    FILE *f = fopen("c-dq.dat.0", "w");
    if (f == NULL)
    {
    printf("Error opening file!\n");
    exit(1);
    }
    for (i = 0; i < cpuData->neq0; i++) {
    for (j = 0; j < cpuData->dq_dim; j++) {
    fprintf(f,"%.10E\n", *(cpuDq + i * cpuData->dq_dim + j));
    }
    }
    fclose(f);

    FILE *f2 = fopen("c-dq.dat.1", "w");
    if (f == NULL)
    {
    printf("Error opening file!\n");
    exit(1);
    }
    for (i = 0; i < gpuData->neq0; i++) {
    for (j = 0; j < gpuData->dq_dim; j++) {
    fprintf(f2,"%.10E\n", *(gpuDq + i * gpuData->dq_dim + j));
    }
    }
    fclose(f2);*/

  free(cpuDq);
  cudaFree(gpuDq_1);
  cudaFree(gpuDq_2);
  cudaFree(gpuDq_3);

  cudaFree(d_iam_1);
  cudaFree(d_jam_1);
  cudaFree(d_res_1);
  cudaFree(d_a_diag_lu_1);
  cudaFree(d_a_off_1);

  cudaFree(d_iam_2);
  cudaFree(d_jam_2);
  cudaFree(d_res_2);
  cudaFree(d_a_diag_lu_2);
  cudaFree(d_a_off_2);

  cudaFree(d_iam_3);
  cudaFree(d_jam_3);
  cudaFree(d_res_3);
  cudaFree(d_a_diag_lu_3);
  cudaFree(d_a_off_3);

  free(cpu_pair);
  free(gpu1_pair);
  free(gpu2_pair);
  free(gpu3_pair);

  freeFeedData(cpuData);
  freeFeedData(gpuData_1);
  freeFeedData(gpuData_2);
  freeFeedData(gpuData_3);
  //_getch();

  return 0;
}

StartEndPair* determineStartEnd(FeedData* data, size_t const &color, size_t const &ipass) {
  StartEndPair* pair = (StartEndPair*) malloc(sizeof(StartEndPair));
  if (color > data->colored_sweeps - 1) {
    pair->start = 1;
    pair->end = 0;
  }
  else {
    switch (ipass) {
    case 1:
      if (data->color_boundary_end[color] == 0) {
	pair->start = 1;
	pair->end = 0;
      }
      else {
	pair->start = data->color_indices[color * 2]; // [0][color]
	pair->end = data->color_boundary_end[color];
      }
      break;
    case 2:
      if (data->color_boundary_end[color] == 0) {
	pair->start = data->color_indices[color * 2];   // [0][color]
	pair->end = data->color_indices[color * 2 + 1]; // [1][color]
      }
      else {
	pair->start = data->color_boundary_end[color] + 1; // [0][color] + 1
	pair->end = data->color_indices[color * 2 + 1];    // [1][color]
      }
      break;
    }
  }
  return pair;
}

void localMalloc(float** dq, size_t dataSize) {
  *dq = (float *)malloc(dataSize);
}

void unifiedMalloc(float** dq, size_t dataSize) {
  cudaMallocManaged(dq, dataSize);
}

__global__ void solveOnGPU(float *gpuDq, double *d_res, int32_t *d_iam, int32_t *d_jam, 
	                   float *d_a_off, double *d_a_diag_lu, size_t start, size_t end, 
	                   int32_t solve_backwards, int32_t nb) {

  __shared__ float fs[5][4];
  __shared__ double a_diag_lu_shared[5][5][4];

  int const k = threadIdx.x % 5;
  int const l = threadIdx.x / 5;
  int n = start + blockIdx.x*blockDim.y + threadIdx.y - 1;

  if (n >= end || l >= 5) return;

  int const istart = d_iam[n];
  int const iend = d_iam[n + 1] - 1;

  float fk;
  double f1, f2, f3, f4, f5;

  // Loop over Non Zeros, 2x unrolled
  fk = 0;
  {
    int j;
    for (j = istart - 1; j < iend - 1; j += 2) {
      fk += A_OFF(k, l, j + 0) * GPU_DQ(l, d_jam[j + 0] - 1);
      fk += A_OFF(k, l, j + 1) * GPU_DQ(l, d_jam[j + 1] - 1);
    }
    for (; j < iend; ++j) {
      fk += A_OFF(k, l, j) * GPU_DQ(l, d_jam[j] - 1);
    }
  }

  // Reduction along the subcolumns, threads with l=0 hold the complete sum
  f1 = fk;
  f1 = f1 + __shfl(fk, k + 1 * 5);
  f1 = f1 + __shfl(fk, k + 2 * 5);
  f1 = f1 + __shfl(fk, k + 3 * 5);
  f1 = f1 + __shfl(fk, k + 4 * 5);

  if (solve_backwards > 0)
    f1 = -RES(k, n) - f1; // assumes solve backward is true
  else
    f1 = RES(k, n) - f1; // assumes solve backward is true

  // Save results of off-diagonal multiplication in shared memory
  if (l == 0) {
    fs[k][threadIdx.y] = f1;
  }

  // Collectively load a_diag_lu into shared memory
  a_diag_lu_shared[k][l][threadIdx.y] = A_DIAG_LU(k, l, n);

  __syncthreads();

  // Redistribute work from all warps to first four threads in the first warp
  n += threadIdx.x;

  if (threadIdx.x < 4 && threadIdx.y == 0 && n < end) {

    // Retrieve data from shared memory
    f1 = fs[0][threadIdx.x];
    f2 = fs[1][threadIdx.x];
    f3 = fs[2][threadIdx.x];
    f4 = fs[3][threadIdx.x];
    f5 = fs[4][threadIdx.x];

    // Forward...sequential access to a_diag_lu

    f2 = f2 - a_diag_lu_shared[1][0][threadIdx.x] * f1;
    f3 = f3 - a_diag_lu_shared[2][0][threadIdx.x] * f1;
    f4 = f4 - a_diag_lu_shared[3][0][threadIdx.x] * f1;
    f5 = f5 - a_diag_lu_shared[4][0][threadIdx.x] * f1;

    f3 = f3 - a_diag_lu_shared[2][1][threadIdx.x] * f2;
    f4 = f4 - a_diag_lu_shared[3][1][threadIdx.x] * f2;
    f5 = f5 - a_diag_lu_shared[4][1][threadIdx.x] * f2;

    f4 = f4 - a_diag_lu_shared[3][2][threadIdx.x] * f3;
    f5 = f5 - a_diag_lu_shared[4][2][threadIdx.x] * f3;

    f5 = ((f5 - a_diag_lu_shared[4][3][threadIdx.x] * f4) *
	  a_diag_lu_shared[4][4][threadIdx.x]);

    // Backward...sequential access to a_diag_lu.

    f1 = f1 - a_diag_lu_shared[0][4][threadIdx.x] * f5;
    f2 = f2 - a_diag_lu_shared[1][4][threadIdx.x] * f5;
    f3 = f3 - a_diag_lu_shared[2][4][threadIdx.x] * f5;
    f4 = ((f4 - a_diag_lu_shared[3][4][threadIdx.x] * f5) *
	  a_diag_lu_shared[3][3][threadIdx.x]);

    f1 = f1 - a_diag_lu_shared[0][3][threadIdx.x] * f4;
    f2 = f2 - a_diag_lu_shared[1][3][threadIdx.x] * f4;
    f3 = ((f3 - a_diag_lu_shared[2][3][threadIdx.x] * f4) *
	  a_diag_lu_shared[2][2][threadIdx.x]);

    f1 = f1 - a_diag_lu_shared[0][2][threadIdx.x] * f3;
    f2 = ((f2 - a_diag_lu_shared[1][2][threadIdx.x] * f3) *
	  a_diag_lu_shared[1][1][threadIdx.x]);

    f1 = ((f1 - a_diag_lu_shared[0][1][threadIdx.x] * f2) *
	  a_diag_lu_shared[0][0][threadIdx.x]);

    GPU_DQ(0, n) = f1;
    GPU_DQ(1, n) = f2;
    GPU_DQ(2, n) = f3;
    GPU_DQ(3, n) = f4;
    GPU_DQ(4, n) = f5;
  }

}


void solveOnCPU(float *cpuDq, FeedData *cpuData, size_t start, size_t end) {

  int32_t j, n, icol, istart, iend, my_index, dq_index;
  float f1, f2, f3, f4, f5;

#pragma omp parallel for default(none)					\
  shared(n, start, end, cpuData, cpuDq)					\
  private(f1, f2, f3, f4, f5, istart, iend, icol, j, my_index, dq_index) schedule(dynamic, 100)
  for (n = start - 1; n < end; ++n) {
    my_index = n * cpuData->nr;
    if (cpuData->solve_backwards > 0) {
      f1 = -cpuData->res[my_index];     // [0][n]
      f2 = -cpuData->res[my_index + 1]; // [1][n]
      f3 = -cpuData->res[my_index + 2]; // [2][n]
      f4 = -cpuData->res[my_index + 3]; // [3][n]
      f5 = -cpuData->res[my_index + 4]; // [4][n]
    }
    else {
      f1 = cpuData->res[my_index];      // [0][n]
      f2 = cpuData->res[my_index + 1];  // [1][n]
      f3 = cpuData->res[my_index + 2];  // [2][n]
      f4 = cpuData->res[my_index + 3];  // [3][n]
      f5 = cpuData->res[my_index + 4];  // [4][n]
    }

    istart = cpuData->iam[n] - 1;
    iend = cpuData->iam[n + 1] - 1;

    for (j = istart; j < iend; ++j) {
      icol = cpuData->jam[j] - 1;
      my_index = j * cpuData->nm * cpuData->nm;
      dq_index = icol * cpuData->dq_dim;
      f1 = f1 - cpuData->a_off[my_index] * cpuDq[dq_index];     
      f2 = f2 - cpuData->a_off[my_index + 1] * cpuDq[dq_index]; 
      f3 = f3 - cpuData->a_off[my_index + 2] * cpuDq[dq_index]; 
      f4 = f4 - cpuData->a_off[my_index + 3] * cpuDq[dq_index]; 
      f5 = f5 - cpuData->a_off[my_index + 4] * cpuDq[dq_index]; 

      my_index = j * cpuData->nm * cpuData->nm + 1 * cpuData->nm;
      dq_index = icol * cpuData->dq_dim + 1;
      f1 = f1 - cpuData->a_off[my_index] * cpuDq[dq_index];     
      f2 = f2 - cpuData->a_off[my_index + 1] * cpuDq[dq_index];          
      f3 = f3 - cpuData->a_off[my_index + 2] * cpuDq[dq_index];  
      f4 = f4 - cpuData->a_off[my_index + 3] * cpuDq[dq_index]; 
      f5 = f5 - cpuData->a_off[my_index + 4] * cpuDq[dq_index];

      my_index = j * cpuData->nm * cpuData->nm + 2 * cpuData->nm;
      dq_index = icol * cpuData->dq_dim + 2;
      f1 = f1 - cpuData->a_off[my_index] * cpuDq[dq_index];
      f2 = f2 - cpuData->a_off[my_index + 1] * cpuDq[dq_index];
      f3 = f3 - cpuData->a_off[my_index + 2] * cpuDq[dq_index];
      f4 = f4 - cpuData->a_off[my_index + 3] * cpuDq[dq_index];
      f5 = f5 - cpuData->a_off[my_index + 4] * cpuDq[dq_index];

      my_index = j * cpuData->nm * cpuData->nm + 3 * cpuData->nm;
      dq_index = icol * cpuData->dq_dim + 3;
      f1 = f1 - cpuData->a_off[my_index] * cpuDq[dq_index];
      f2 = f2 - cpuData->a_off[my_index + 1] * cpuDq[dq_index];
      f3 = f3 - cpuData->a_off[my_index + 2] * cpuDq[dq_index];
      f4 = f4 - cpuData->a_off[my_index + 3] * cpuDq[dq_index];
      f5 = f5 - cpuData->a_off[my_index + 4] * cpuDq[dq_index];

      my_index = j * cpuData->nm * cpuData->nm + 4 * cpuData->nm;
      dq_index = icol * cpuData->dq_dim + 4;
      f1 = f1 - cpuData->a_off[my_index] * cpuDq[dq_index];
      f2 = f2 - cpuData->a_off[my_index + 1] * cpuDq[dq_index];
      f3 = f3 - cpuData->a_off[my_index + 2] * cpuDq[dq_index];
      f4 = f4 - cpuData->a_off[my_index + 3] * cpuDq[dq_index];
      f5 = f5 - cpuData->a_off[my_index + 4] * cpuDq[dq_index];

    } // end istart loop


    // Forward...sequential access to a_diag_lu.
    my_index = n * cpuData->nm * cpuData->nm;
    f2 = f2 - cpuData->a_diag_lu[my_index + 1] * f1;
    f3 = f3 - cpuData->a_diag_lu[my_index + 2] * f1;
    f4 = f4 - cpuData->a_diag_lu[my_index + 3] * f1;
    f5 = f5 - cpuData->a_diag_lu[my_index + 4] * f1;

    my_index = n * cpuData->nm * cpuData->nm + 1 * cpuData->nm;
    f3 = f3 - cpuData->a_diag_lu[my_index + 2] * f2;
    f4 = f4 - cpuData->a_diag_lu[my_index + 3] * f2;
    f5 = f5 - cpuData->a_diag_lu[my_index + 4] * f2;

    my_index = n * cpuData->nm * cpuData->nm + 2 * cpuData->nm;
    f4 = f4 - cpuData->a_diag_lu[my_index + 3] * f3;
    f5 = f5 - cpuData->a_diag_lu[my_index + 4] * f3;

    my_index = n * cpuData->nm * cpuData->nm + 3 * cpuData->nm;
    f5 = f5 - cpuData->a_diag_lu[my_index + 4] * f4;

    // Backward...sequential access to a_diag_lu.
    my_index = n * cpuData->nm * cpuData->nm + 4 * cpuData->nm;
    dq_index = n * cpuData->dq_dim + 4;
    cpuDq[dq_index] = f5 * cpuData->a_diag_lu[my_index + 4];
    f1 = f1 - cpuData->a_diag_lu[my_index] * cpuDq[dq_index];
    f2 = f2 - cpuData->a_diag_lu[my_index + 1] * cpuDq[dq_index];
    f3 = f3 - cpuData->a_diag_lu[my_index + 2] * cpuDq[dq_index];
    f4 = f4 - cpuData->a_diag_lu[my_index + 3] * cpuDq[dq_index];

    my_index = n * cpuData->nm * cpuData->nm + 3 * cpuData->nm;
    dq_index = n * cpuData->dq_dim + 3;
    cpuDq[dq_index] = f4 * cpuData->a_diag_lu[my_index + 3];
    f1 = f1 - cpuData->a_diag_lu[my_index] * cpuDq[dq_index];
    f2 = f2 - cpuData->a_diag_lu[my_index + 1] * cpuDq[dq_index];
    f3 = f3 - cpuData->a_diag_lu[my_index + 2] * cpuDq[dq_index];

    my_index = n * cpuData->nm * cpuData->nm + 2 * cpuData->nm;
    dq_index = n * cpuData->dq_dim + 2;
    cpuDq[dq_index] = f3 * cpuData->a_diag_lu[my_index + 2];
    f1 = f1 - cpuData->a_diag_lu[my_index] * cpuDq[dq_index];
    f2 = f2 - cpuData->a_diag_lu[my_index + 1] * cpuDq[dq_index];

    my_index = n * cpuData->nm * cpuData->nm + 1 * cpuData->nm;
    dq_index = n * cpuData->dq_dim + 1;
    cpuDq[dq_index] = f2 * cpuData->a_diag_lu[my_index + 1];
    f1 = f1 - cpuData->a_diag_lu[my_index] * cpuDq[dq_index];

    my_index = n * cpuData->nm * cpuData->nm;
    dq_index = n * cpuData->dq_dim;
    cpuDq[dq_index] = f1 * cpuData->a_diag_lu[my_index];

  } // end parallel loop
}

void startDataTransfer(float* dq, size_t row, size_t col, SendRecvType* sr_loc) {
  int32_t senddim, recvdim, inde, inode, l;
  senddim = sr_loc->sendproc[4] - 1;
  recvdim = sr_loc->recvproc[4] - 1;
  //printf("senddim = %d -- recvdim = %d\n", senddim, recvdim);
  //char np[2];
  // check if work array is allocated and the correct size
  if (!sr_loc->single_mat_allocated) {
    sr_loc->single_mat_recvdata = (float *)malloc(max_val(row, 1) * max_val(recvdim, 1) 
						  * sizeof(*sr_loc->single_mat_recvdata));
    sr_loc->single_mat_senddata = (float *)malloc(max_val(row, 1) * max_val(senddim, 1) 
						  * sizeof(*sr_loc->single_mat_senddata));
    sr_loc->single_mat_allocated = 1;
  }
  for (inde = 0; inde < senddim; inde++)
    {
      inode = sr_loc->sendindex[inde] - 1;
      for (l = 0; l < row; l++)
	{
	  //printf("send inode, l, inde, val =  %d\t%d\t%d\t%.5e\n", inode, l, inde, 
	  //dq[inode * row + l]);
	  //scanf("%s", np);
	  sr_loc->single_mat_senddata[inde * row + l] = dq[inode * row + l];
	}
    }
}

void completeDataTransfer(float* dq, size_t row, size_t col, SendRecvType* sr_loc, 
			  float* buffer[], int32_t* sendproc[], size_t const &rank) {
  int32_t recvdim, inde, inode, l, senddim, send_index;
  //size_t i;
  float* single_mat_recvdata = NULL;
  for (size_t i = 0; i < 4; i++)
    {
      recvdim = sr_loc->recvproc[i+1] - 1;
      inde = sr_loc->recvproc[i] - 1;
      senddim = sendproc[i][rank + 1] - 1;
      send_index = sendproc[i][rank] - 1;
      single_mat_recvdata = buffer[i];
      char np[2];
      //printf("i = %d, start = %d, end = %d", i, inde, recvdim);
      //printf("i = %d, start = %d, end = %d", i, send_index, senddim);
      //scanf("%s", np);
      for (; inde < recvdim; inde++, send_index++)
	{
	  inode = sr_loc->recvindex[inde] - 1;
	  for (l = 0; l < row; l++)
	    {
	      //printf("recive inode, l, inde, val =  %d\t%d\t%d\t%.5e\n", inode, l, inde, 
	      //single_mat_recvdata[send_index * row + l]);
	      //scanf("%s", np);
	      dq[inode * row + l] = single_mat_recvdata[send_index * row + l];
	    }
	}
      single_mat_recvdata = NULL;
    }
}

void load_data_on_gpu(FeedData *gpuData, int32_t **d_iam, int32_t **d_jam, double **d_res, 
		      double **d_a_diag_lu, float ** d_a_off) {
  cudaError_t error;
  int32_t dataSize;

  // allocate and copy gpu memory for iam
  dataSize = gpuData->nia * sizeof(**d_iam);
  error = cudaMalloc((void **)d_iam, dataSize);
  if (error != cudaSuccess) {
    printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }
  error = cudaMemcpy(*d_iam, gpuData->iam, dataSize, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }

  // allocate and copy gpu memory for jam
  dataSize = gpuData->nja * sizeof(**d_jam);
  error = cudaMalloc((void **)d_jam, dataSize);
  if (error != cudaSuccess) {
    printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }
  error = cudaMemcpy(*d_jam, gpuData->jam, dataSize, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }

  // allocate and copy gpu memory for res
  dataSize = gpuData->nr * gpuData->neq0 * sizeof(**d_res);
  error = cudaMalloc((void **)d_res, dataSize);
  if (error != cudaSuccess) {
    printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }
  error = cudaMemcpy(*d_res, gpuData->res, dataSize, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }

  // allocate and copy gpu memory for a_diag_lu
  dataSize = gpuData->nm * gpuData->nm * gpuData->neq0 * sizeof(**d_a_diag_lu);
  error = cudaMalloc((void **)d_a_diag_lu, dataSize);
  if (error != cudaSuccess) {
    printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }
  error = cudaMemcpy(*d_a_diag_lu, gpuData->a_diag_lu, dataSize, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }

  // allocate memory for a_off
  dataSize = gpuData->nm * gpuData->nm * gpuData->nja * sizeof(**d_a_off);
  error = cudaMalloc((void **)d_a_off, dataSize);
  if (error != cudaSuccess) {
    printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }
  error = cudaMemcpy(*d_a_off, gpuData->a_off, dataSize, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
	   error, __LINE__);
    exit(EXIT_FAILURE);
  }
}

