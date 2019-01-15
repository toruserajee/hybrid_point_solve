#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "load_data.h"

void reverse_bytes(void *data, size_t size)
{
	char *i, *j;
	char tmp;
	for (i = (char*)data, j = i + size - 1; i < j; i++, j--) {
		tmp = *i;
		*i = *j;
		*j = tmp;
	}
}

void revArrInt(int32_t* arr, size_t start, size_t end)
{
	int32_t temp;
	while (start < end)
	{
		temp = arr[start];
		arr[start] = arr[end];
		arr[end] = temp;
		start++;
		end--;
	}
}

void revArrFloat(float* arr, size_t start, size_t end)
{
	float temp;
	while (start < end)
	{
		temp = arr[start];
		arr[start] = arr[end];
		arr[end] = temp;
		start++;
		end--;
	}
}

void revArrDouble(double* arr, size_t start, size_t end)
{
	double temp;
	while (start < end)
	{
		temp = arr[start];
		arr[start] = arr[end];
		arr[end] = temp;
		start++;
		end--;
	}
}

void extract(FILE* fp, void *data, size_t size)
{
	fread(data, size, 1, fp);
	reverse_bytes(data, size);
}

size_t max_t(size_t a, size_t b) {
	if (a > b) return a;
	else return b;
}


float* load_data(char* datafile, FeedData* data, void(*mallocFunc)(float**, size_t))
{
	float *dq = NULL;
	int32_t i, j;
	FILE* fp = fopen(datafile, "rb");

	extract(fp, &data->colored_sweeps, sizeof(data->colored_sweeps));
	extract(fp, &data->max_colored_sweeps, sizeof(data->max_colored_sweeps));
	extract(fp, &data->neq0, sizeof(data->neq0));
	extract(fp, &data->nia, sizeof(data->nia));
	extract(fp, &data->nja, sizeof(data->nja));
	extract(fp, &data->n_eqns, sizeof(data->n_eqns));
	extract(fp, &data->dq_dim, sizeof(data->dq_dim));
	extract(fp, &data->nr, sizeof(data->nr));
	extract(fp, &data->nm, sizeof(data->nm));
	extract(fp, &data->n_sweeps, sizeof(data->n_sweeps));
	extract(fp, &data->neqmax, sizeof(data->neqmax));
	extract(fp, &data->nb, sizeof(data->nb));
	extract(fp, &data->solve_backwards, sizeof(data->solve_backwards));
	extract(fp, &data->outer_sweeps, sizeof(data->outer_sweeps));
	extract(fp, &data->sr_size, sizeof(data->sr_size));
	printf("Number of block 5x5 equations in data file : %d\n", data->neq0);
	// allocate memory for color_indices
	size_t dataSize = 2 * data->colored_sweeps * sizeof(*data->color_indices);
	data->color_indices = (int32_t *) malloc(dataSize);
	extract(fp, data->color_indices, dataSize);
	revArrInt(data->color_indices, 0, 2 * data->colored_sweeps - 1);
	// allocate memory for color_boundary_end
	dataSize = data->colored_sweeps * sizeof(*data->color_boundary_end);
	data->color_boundary_end = (int32_t *) malloc(dataSize);
	extract(fp, data->color_boundary_end, dataSize);
	revArrInt(data->color_boundary_end, 0, data->colored_sweeps - 1);
	// allocate memory for iam
	dataSize = data->nia * sizeof(*data->iam);
	data->iam = (int32_t *) malloc(dataSize);
	extract(fp, data->iam, dataSize);
	revArrInt(data->iam, 0, data->nia - 1);
	// allocate memory for jam
	dataSize = data->nja * sizeof(*data->jam);
	data->jam = (int32_t *) malloc(dataSize);
	extract(fp, data->jam, dataSize);
	revArrInt(data->jam, 0, data->nja - 1);
	// allocate memory for res
	dataSize = data->nr * data->neq0 * sizeof(*data->res);
	data->res = (double *) malloc(dataSize);
	extract(fp, data->res, dataSize);
	revArrDouble(data->res, 0, data->nr * data->neq0 - 1);
	// allocate memory for dq
	dataSize = data->dq_dim * data->neqmax * sizeof(*dq);
	mallocFunc(&dq, dataSize);
	//dq = (float *)malloc(dataSize); 
	//cudaMallocManaged(&dq, dataSize);
	extract(fp, dq, dataSize);
	revArrFloat(dq, 0, data->dq_dim * data->neqmax - 1);
	// allocate memory for a_diag_lu
	dataSize = data->nm * data->nm * data->neq0 * sizeof(*data->a_diag_lu);
	data->a_diag_lu = (double *) malloc(dataSize);
	extract(fp, data->a_diag_lu, dataSize);
	revArrDouble(data->a_diag_lu, 0, data->nm * data->nm * data->neq0 - 1);
	//printf("a_diag_lu = %.10e\t%.10e\n", data->a_diag_lu[0], data->a_diag_lu[data->nm * data->nm * data->neq0 - 1]);
	// allocate memory for a_off
	dataSize = data->nm * data->nm * data->nja * sizeof(*data->a_off);
	data->a_off = (float *) malloc(dataSize);
	extract(fp, data->a_off, dataSize);
	revArrFloat(data->a_off, 0, data->nm * data->nm * data->nja - 1);
	//printf("a_off = %.10e\t%.10e\n", data->a_off[0], data->a_off[data->nm * data->nm * data->nja - 1]);
	// allocate memory for sr
	dataSize = data->sr_size * sizeof(*data->sr);
	data->sr = (SendRecvType *)malloc(dataSize);
	SendRecvType* srStrcut = NULL;
	for (i = 0; i < data->sr_size; ++i)
	{
		extract(fp, &data->dim1, sizeof(data->dim1));
		extract(fp, &data->dim2, sizeof(data->dim2));
		extract(fp, &data->dim3, sizeof(data->dim3));
		extract(fp, &data->dim4, sizeof(data->dim4));
		extract(fp, &data->dim5, sizeof(data->dim5));
		extract(fp, &data->dim6, sizeof(data->dim6));
		extract(fp, &data->dim7, sizeof(data->dim7));
		extract(fp, &data->dim8, sizeof(data->dim8));
		extract(fp, &data->dim9, sizeof(data->dim9));
		extract(fp, &data->dim10, sizeof(data->dim10));
		//printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", data->dim1, data->dim2, data->dim3, data->dim4, data->dim5, data->dim6, data->dim7, data->dim8, data->dim9, data->dim10);
		srStrcut = (SendRecvType *) malloc(sizeof(SendRecvType));
		srStrcut->sendproc = (int32_t *) malloc(max_t(data->dim1, 1) * sizeof(*srStrcut->sendproc));
		srStrcut->recvproc = (int32_t *) malloc(max_t(data->dim2, 1) * sizeof(*srStrcut->recvproc));
		srStrcut->sendindex = (int32_t *) malloc(max_t(data->dim3, 1) * sizeof(*srStrcut->sendindex));
		srStrcut->recvindex = (int32_t *) malloc(max_t(data->dim4, 1) * sizeof(*srStrcut->recvindex));
		srStrcut->double_mat_senddata = (double *) malloc(max_t(data->dim5, 1) * max_t(data->dim6, 1) * sizeof(*srStrcut->double_mat_senddata));
		srStrcut->double_mat_recvdata = (double *) malloc(max_t(data->dim7, 1) * max_t(data->dim8, 1) * sizeof(*srStrcut->double_mat_recvdata));
		srStrcut->array_of_statuses = (int32_t *) malloc(max_t(data->dim9, 1) * max_t(data->dim10, 1) * sizeof(*srStrcut->array_of_statuses));
		for (j = 0; j < data->dim1; ++j)
			extract(fp, &srStrcut->sendproc[j], sizeof(srStrcut->sendproc[j]));

		for (j = 0; j < data->dim2; ++j)
			extract(fp, &srStrcut->recvproc[j], sizeof(srStrcut->recvproc[j]));

		for (j = 0; j < data->dim3; ++j) 
			extract(fp, &srStrcut->sendindex[j], sizeof(srStrcut->sendindex[j]));

		for (j = 0; j < data->dim4; ++j)
			extract(fp, &srStrcut->recvindex[j], sizeof(srStrcut->recvindex[j]));

		srStrcut->double_mat_allocated = 1; //true
		srStrcut->single_mat_allocated = 0; //false
		data->sr[i] = *srStrcut;
		srStrcut = NULL;
	}
	extract(fp, &data->golden, sizeof(data->golden));
	//printf("golden %.30e\n", data->golden);

	fclose(fp);
	return dq;
}

void freeFeedData(FeedData* data)
{
	int32_t i;

	if (data != NULL)
	{
		if (data->color_indices != NULL)
			free(data->color_indices);

		if (data->color_boundary_end != NULL)
			free(data->color_boundary_end);

		if (data->iam != NULL)
			free(data->iam);

		if (data->jam != NULL)
			free(data->jam);

		if (data->res != NULL)
			free(data->res);

		if (data->a_diag_lu != NULL)
			free(data->a_diag_lu);

		if (data->a_off != NULL)
			free(data->a_off);

		if (data->sr != NULL)
		{
			for (i = 0; i < data->sr_size; ++i)
			{
				if (data->sr[i].sendproc != NULL)
					free(data->sr[i].sendproc);

				if (data->sr[i].recvproc != NULL)
					free(data->sr[i].recvproc);

				if (data->sr[i].sendindex != NULL)
					free(data->sr[i].sendindex);

				if (data->sr[i].recvindex != NULL)
					free(data->sr[i].recvindex);

				if (data->sr[i].double_mat_senddata != NULL)
					free(data->sr[i].double_mat_senddata);

				if (data->sr[i].double_mat_recvdata != NULL)
					free(data->sr[i].double_mat_recvdata);

				if (data->sr[i].array_of_statuses != NULL)
					free(data->sr[i].array_of_statuses);
			}
			free(data->sr);
		}

		free(data);
	}
}

