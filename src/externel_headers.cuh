/*
 * externel_headers.cuh
 *
 *  Created on: Jun 1, 2018
 *      Author: hondo
 */

#ifndef EXTERNEL_HEADERS_CUH_
#define EXTERNEL_HEADERS_CUH_

#include <cuda.h>

#define HANDLE_CUDA_ERROR(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

/** place all enums here **/
typedef enum LEARNING_PROBLEM {regression,classification} LEARNING_PROBLEM;



#endif /* EXTERNEL_HEADERS_CUH_ */
