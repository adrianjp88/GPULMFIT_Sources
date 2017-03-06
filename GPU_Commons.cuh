#ifndef _GPU_COMMONS_CUH_
#define _GPU_COMMONS_CUH_

#include "cuda_runtime_api.h" // For CUDA
#include "math.h"

/* Conditional Compilation Constants */
#define _MATLAB_DISPLAY_CUDA_SETTING_INFO_ // Display the CUDA configuration

/* Macros for CUDA threads and blocks indices */
#define tidx threadIdx.x
#define bidx blockIdx.x
#define bdim blockDim.x
#define gdim gridDim.x

/* Specify the functions for single precision data type */
#define cuFABS fabsf
#define cuSQRT sqrtf
#define cuRSQRT rsqrtf
#define cuEXP expf
#define cuPOW powf
#define cuCEIL ceilf

#define log2f(x) (logf(x+0.0f)/logf(2.0f))
#define nearest_pow2(x) (powf(2.0f, floorf(log2f(x))))

/* Utility macros */
#ifndef _CUTIL_INLINE_FUNCTIONS_RUNTIME_H_ // To avoid the definition confliction in cutil_inline.h
	#define MIN(a, b)((a)<(b)?(a):(b))
	#define MAX(a, b)((a)>(b)?(a):(b))
#endif
#define SIGN(a)((a)>0?1:-1)

/* **********************************************************************************************************
		                                                              Commonly used functions 
*************************************************************************************************************/
extern bool CheckCudaError(cudaError, char *, char *);
extern bool CuDeviceInit(struct cudaDeviceProp *, int, char *);
extern bool Get_Kernel_Basic_Config(int, struct cudaDeviceProp, struct cudaFuncAttributes, int *, size_t *, char *);

#endif //_GPU_COMMONS_H_