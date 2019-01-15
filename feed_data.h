#include <stdint.h>
#ifndef FEED_STRUCT_H
#define FEED_STRUCT_H
#include "interp_defs.h"

struct FeedData {

	int32_t colored_sweeps;
	int32_t max_colored_sweeps;
	int32_t neq0;
	int32_t nia;
	int32_t nja;
	int32_t n_eqns;
	int32_t dq_dim;
	int32_t nr;
	int32_t nm;
	int32_t n_sweeps;
	int32_t neqmax;
	int32_t nb;
	int32_t solve_backwards;
	int32_t outer_sweeps;
	int32_t sr_size;
	int32_t* iam;
	int32_t* jam;
	int32_t* color_boundary_end;
	int32_t* color_indices; //2d Array
	double golden;
	double* res; //2d Array
	double* a_diag_lu; //3dArray
	float* a_off; //3dArray
	SendRecvType* sr;
	int32_t dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10;

};

typedef struct FeedData FeedData;
#endif

