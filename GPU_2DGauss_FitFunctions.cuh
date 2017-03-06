#ifndef _GPU_2DGAUSS_FITFUNCTION_CUH_
#define _GPU_2DGAUSS_FITFUNCTION_CUH_

#include "GPU_Commons.cuh" 

#define float float

/*	
	Basic parameters passed to both the fitting function and the analytical Jacobian function 
	through the third argument of both functions.
*/
struct GPU_FUNC_CONSTS_STRUCT{
	int sv_n;				// Number of fitting parameters.
	int sv_m;				// Number of data points.
	int sv_ImgDim;		// Image dimension size (assume square images).
	int sv_JacMethod;	// 0 - Analytical Jacobian; 1 - Numerical Jacobian.	
	int sv_FitMethod;		// 0 - maximum likelihood estimator, or 1 - unweighted least squares. 
	float *sp_CurrData;	// The pointer to the data set of the image which is fitting. 
	float *sp_buffer;		// The pointer to a temporary buffer. 
};
typedef struct GPU_FUNC_CONSTS_STRUCT GPU_FUNC_CONSTS;

/* External function prototype - must follow the instruction in GPU_LMFit.cuh */
extern __device__ int GPU_FitFunction(const float *, float *, void *);
extern __device__ int GPU_AnalyticalJacobian(const float *, float *, void *);

#endif //_GPU_2DGAUSS_FITFUNCTION_CUH_