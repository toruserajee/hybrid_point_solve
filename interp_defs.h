#ifndef INTERP_DEFS_H
#define INTERP_DEFS_H
#include <stdint.h>

struct sendrecv_type {
	int32_t* sendproc;
	int32_t* recvproc;
	int32_t* sendindex;
	int32_t* recvindex;
	double* double_mat_senddata;   //2dArray 
	double* double_mat_recvdata;   //2dArray 
	int32_t* array_of_statuses;    //2dArray
	uint8_t double_mat_allocated;
	float* single_mat_recvdata;    //2dArray
	float* single_mat_senddata;    //2dArray
	uint8_t single_mat_allocated;
};
typedef struct sendrecv_type SendRecvType;
#endif // #ifndef INTERP_DEFS_H
