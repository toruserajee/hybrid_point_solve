#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include <omp.h>

#include "load_data.h"

using namespace std;

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
void solveOnCPU(float *cpuDq, FeedData *cpuData, size_t start, size_t end);
void startDataTransfer(float* dq, size_t row, size_t col, SendRecvType* sr_loc);
void completeDataTransfer(float* dq, size_t row, size_t col, SendRecvType* sr_loc, 
			  float* buffer[], int32_t* sendproc[], size_t const &rank);
StartEndPair* determineStartEnd(FeedData *data, size_t const &color, size_t const &ipass);

double TimeSpecToSeconds(struct timespec* ts) {
  return (double)ts->tv_sec + (double)ts->tv_nsec/1000000000.0;
}

int main() {

  int32_t i, j;
  int32_t sweep, ipass;
  StartEndPair *cpu0_pair = NULL, *cpu1_pair = NULL, *cpu2_pair = NULL, *cpu3_pair = NULL;
  int32_t color, loop_colored_sweeps;
  int32_t sweep_start, sweep_end, sweep_stride;
  FeedData *cpuData_0, *cpuData_1, *cpuData_2, *cpuData_3;
  cpuData_0 = (FeedData *)malloc(sizeof(FeedData));
  cpuData_1 = (FeedData *)malloc(sizeof(FeedData));
  cpuData_2 = (FeedData *)malloc(sizeof(FeedData));
  cpuData_3 = (FeedData *)malloc(sizeof(FeedData));
  float *cpuDq_0, *cpuDq_1, *cpuDq_2, *cpuDq_3;

  float* buffer[4];
  int32_t* sendproc[4];

  double wtime1, wtime2, totalTransferTime = 0.0f;

  printf("Loading Data ... \n");
#pragma omp parallel
  {
#pragma omp single nowait
    {
      cpuDq_0 = load_data("4-linear_system.dat.0", cpuData_0, localMalloc);
    }
#pragma omp single nowait
    {
      cpuDq_1 = load_data("4-linear_system.dat.1", cpuData_1, localMalloc);
    }
#pragma omp single nowait
    {
      cpuDq_2 = load_data("4-linear_system.dat.2", cpuData_2, localMalloc);
    }
#pragma omp single nowait
    {
      cpuDq_3 = load_data("4-linear_system.dat.3", cpuData_3, localMalloc);
    }
  }
#pragma omp barrier

  printf("Data Loaded successfully... \n");
  printf("Solving Ax=b...\n");
  int32_t outer_sweeps = cpuData_0->outer_sweeps;
  struct timespec begin;
  clock_gettime(CLOCK_MONOTONIC, &begin);

  for (i = 0; i < outer_sweeps; ++i)
    {

      loop_colored_sweeps = cpuData_0->max_colored_sweeps;

      sweep_start = 0;
      sweep_end = loop_colored_sweeps;
      sweep_stride = 1;

      if (cpuData_0->solve_backwards > 1 || cpuData_0->solve_backwards < -1) {
	sweep_start = loop_colored_sweeps - 1;
	sweep_end = -1;
	sweep_stride = -1;
      }

      if (cpuData_0->n_eqns <= 0) {
	sweep_start = 0;
	sweep_end = 1;
	sweep_stride = -1;
      }

      for (sweep = 0; sweep < cpuData_0->n_sweeps; ++sweep) { // sweeping
	for (color = sweep_start; color < sweep_end; color += sweep_stride) { // color_sweeps
	  for (ipass = 1; ipass <= 2; ++ipass) { // ipass

	    cpu0_pair = determineStartEnd(cpuData_0, color, ipass);
	    solveOnCPU(cpuDq_0, cpuData_0, cpu0_pair->start, cpu0_pair->end);
	    cpu1_pair = determineStartEnd(cpuData_1, color, ipass);
	    solveOnCPU(cpuDq_1, cpuData_1, cpu1_pair->start, cpu1_pair->end);
	    cpu2_pair = determineStartEnd(cpuData_2, color, ipass);
	    solveOnCPU(cpuDq_2, cpuData_2, cpu2_pair->start, cpu2_pair->end);
	    cpu3_pair = determineStartEnd(cpuData_3, color, ipass);
	    solveOnCPU(cpuDq_3, cpuData_3, cpu3_pair->start, cpu3_pair->end);

            #pragma omp barrier
	    switch (ipass) {
	    case 1:
	      {
		wtime1 = omp_get_wtime();
                #pragma omp parallel
		{
                  #pragma omp single nowait
		  {
		    startDataTransfer(cpuDq_0, cpuData_0->dq_dim, cpuData_0->neqmax, 
				      &cpuData_0->sr[color]);
		  }

                  #pragma omp single nowait
		  {
		    startDataTransfer(cpuDq_1, cpuData_1->dq_dim, cpuData_1->neqmax, 
				      &cpuData_1->sr[color]);
		  }

                  #pragma omp single nowait
		  {
		    startDataTransfer(cpuDq_2, cpuData_2->dq_dim, cpuData_2->neqmax, 
				      &cpuData_2->sr[color]);
		  }

                  #pragma omp single nowait
		  {
		    startDataTransfer(cpuDq_3, cpuData_3->dq_dim, cpuData_3->neqmax, 
				      &cpuData_3->sr[color]);
		  }
		}
               #pragma omp barrier
		wtime1 = omp_get_wtime() - wtime1;
		break;
	}
	case 2:
	  {
	  buffer[0] = cpuData_0->sr[color].single_mat_senddata;
	  buffer[1] = cpuData_1->sr[color].single_mat_senddata;
	  buffer[2] = cpuData_2->sr[color].single_mat_senddata;
	  buffer[3] = cpuData_3->sr[color].single_mat_senddata;
	  sendproc[0] = cpuData_0->sr[color].sendproc;
	  sendproc[1] = cpuData_1->sr[color].sendproc;
	  sendproc[2] = cpuData_2->sr[color].sendproc;
	  sendproc[3] = cpuData_3->sr[color].sendproc;
	  wtime2 = omp_get_wtime();

                #pragma omp parallel //shared(buffer, sendproc)
		{
                  #pragma omp single nowait
		  {
		    completeDataTransfer(cpuDq_0, cpuData_0->dq_dim, cpuData_0->neqmax, 
		                         &cpuData_0->sr[color], buffer, sendproc, 0);
		  }

                  #pragma omp single nowait
		  {
		    completeDataTransfer(cpuDq_1, cpuData_1->dq_dim, cpuData_1->neqmax, 
	                                 &cpuData_1->sr[color], buffer, sendproc, 1);
		  }

                  #pragma omp single nowait
 		  {
		    completeDataTransfer(cpuDq_2, cpuData_2->dq_dim, cpuData_2->neqmax, 
		                         &cpuData_2->sr[color], buffer, sendproc, 2);
		  }

                  #pragma omp single nowait
		  {
	            completeDataTransfer(cpuDq_3, cpuData_3->dq_dim, cpuData_3->neqmax, 
	                                 &cpuData_3->sr[color], buffer, sendproc, 3);
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
  for (i = 0; i < cpuData_0->neq0; i++) {
    for (j = 0; j < cpuData_0->dq_dim; j++) {
      rms1 = rms1 + pow(*(cpuDq_0 + i * cpuData_0->dq_dim + j), 2);
    }
  }

  rms2 = 0.0;
  for (i = 0; i < cpuData_1->neq0; i++) {
    for (j = 0; j < cpuData_1->dq_dim; j++) {
      rms2 = rms2 + pow(*(cpuDq_1 + i * cpuData_1->dq_dim + j), 2);
    }
  }

  rms3 = 0.0;
  for (i = 0; i < cpuData_2->neq0; i++) {
    for (j = 0; j < cpuData_2->dq_dim; j++) {
      rms3 = rms3 + pow(*(cpuDq_2 + i * cpuData_2->dq_dim + j), 2);
    }
  }

  rms4 = 0.0;
  for (i = 0; i < cpuData_3->neq0; i++) {
    for (j = 0; j < cpuData_3->dq_dim; j++) {
      rms4 = rms4 + pow(*(cpuDq_3 + i * cpuData_3->dq_dim + j), 2);
    }
  }

  rms_sum = rms1 + rms2 + rms3 + rms4;
  neq = cpuData_0->neq0 + cpuData_1->neq0 + cpuData_2->neq0 + cpuData_3->neq0;
  rms_sum = sqrt(rms_sum / (double)neq / (double)cpuData_0->nb);

  if (isnan(rms_sum) || (rms_sum - cpuData_0->golden) >= tol) {
    printf("test failed: %.10E", rms_sum - cpuData_0->golden);
  }
  else {
    printf("test passed, result = %.10E\tgolden = %.10E\n", rms_sum, cpuData_0->golden);
  }
	
  free(cpuDq_0);
  free(cpuDq_1);
  free(cpuDq_2);
  free(cpuDq_3);

  free(cpu0_pair);
  free(cpu1_pair);
  free(cpu2_pair);
  free(cpu3_pair);

  freeFeedData(cpuData_0);
  freeFeedData(cpuData_1);
  freeFeedData(cpuData_2);
  freeFeedData(cpuData_3);
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

