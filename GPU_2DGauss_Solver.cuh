#ifndef _GPU_2DGAUSS_SOLVER_H_
#define _GPU_2DGAUSS_SOLVER_H_

// GPU_LMFit library
#include "GPU_LMFit.cuh"
#include "GPU_LMFit_Accessories.h"

/* Kernel constant inputs struct definition */
struct TWODGAUSS_GPU_LMFIT_INS_STRUCT {
	int n;								// Number of fitting parameters.
	int m;								// Number of data points.
	int ImgDim;						// Image dimension size (assume square images).
	int JacMethod;					// 0 - Analytical Jacobian; 1 - Numerical Jacobian.	
	int FitMethod;					// 0 - maximum likelihood estimator, or 1 - unweighted least squares. 
	// GPU temporary buffers - required for work variables in GPU-LMFit 
	float *GPU_LMFit_Real_Mem;
	// Additional parameters - required for GPU-LMFit to determine how many n- or m-length vectors can be in shared memory. 
	int NumOfGPULMFitSharednVec;
	int NumOfGPULMFitSharedmVec;
};
typedef struct TWODGAUSS_GPU_LMFIT_INS_STRUCT TWODGAUSS_GPU_LMFIT_INS;


/* External function prototypes */
extern int GPU_LMFIT_Solver(int, int, int, int, int, int, float *, int, float *,  int *, int *, float *, float *, float *,char *);


#endif //_GPU_2DGAUSS_SOLVER_H_
