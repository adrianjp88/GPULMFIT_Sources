#include "Apps.cuh"

#include "GPU_2DGauss_Solver.cuh"
#include "GPU_2DGauss_FitFunctions.cuh"

#include <float.h>

#if defined(_MATLAB_DISPLAY_CUDA_SETTING_INFO_) 
#include "D:\Program Files (x86)\MATLAB\R2014b\extern\include\mex.h"
#endif

/* Conditional Compilation Constants */

/* 2*sizeof(float) for sp_smax and sp_smin, while 1*sizeof(int) for sp_smax_idx. */
#define GPU_2DGau_Init_Extern_Shared_Mem_Size_Per_Thread (2*(2*sizeof(float)+1*sizeof(int)))

/* Variables in GPU constant memory */
__constant__ LMFIT_CRITERIA GPU_2DGaussFit_LM_Config = {1e-4f, 1e-4f, 1e-5f, 1.0e-07f, 20.0f, 50}; // Good enough for GPU 2D Gauss fit (defalut)
//__constant__ LMFIT_CRITERIA GPU_2DGaussFit_LM_Config = { 1e-4f, FLT_EPSILON, FLT_EPSILON, 0.f, 0.f, 20 }; // 

/* External shared memory pointer used by all kernels */
extern __shared__ char SVec[]; 


/***********************************************************************************************************
		                                                               Device  __global__ functions 
*************************************************************************************************************/

__global__ void TwoDGauss_GPU_LMFit_Kernel(TWODGAUSS_GPU_LMFIT_INS GPU_LMFit_Ins, float *x, 
	float *GPUFitDataBuffer, int *InfoNum, float *Chisq, int SVecSize, unsigned long long GPU_LMFit_Info)
{
	/*
	Function: perform two-dimensional Gaussian fit using GPU_LMFit
	Parameters description:
		GPU_LMFit_Ins (input) - some basic parameters, see the type definition in GPU_2DGauss_Solver.cuh;
		x (input and output) - the fitting parameters array in global memory of GPU;
		GPUImgDataBuffer (input) - the images data in global memory of GPU;
		InfoNum (output) - if it is not NULL, it returns the exist code from GPU_LMFit ;
		Chisq (output) - if it is not NULL, it returns Chi square of the fit;
		SVecSize (input) - the size of externally allocated shared memory for each CUDA block;
		GPU_LMFit_Info (input) - information required by GPU_LMFit (see the function Get_GPU_LMFit_Kernel_Config).

	Note:
		Each block completes one fit, so both TwoDGauss_GPU_LMFit_Kernel & GPU_2DGau_Init_Kernel 
		must use the same gdim. 
	*/

	/* LMFit critieria */
	__shared__ LMFIT_CRITERIA GPU_LM_Config; 

	/*Fit Function variabls */
	__shared__ GPU_FUNC_CONSTS GPU_f_Cs;

	if(tidx==0){ 
		GPU_LM_Config = GPU_2DGaussFit_LM_Config;
		GPU_f_Cs.sv_n = GPU_LMFit_Ins.n;
		GPU_f_Cs.sv_m = GPU_LMFit_Ins.m;
		GPU_f_Cs.sv_ImgDim = GPU_LMFit_Ins.ImgDim;
		GPU_f_Cs.sv_JacMethod = GPU_LMFit_Ins.JacMethod;
		GPU_f_Cs.sv_FitMethod = GPU_LMFit_Ins.FitMethod;
		GPU_f_Cs.sp_CurrData = &GPUFitDataBuffer[bidx*GPU_LMFit_Ins.m];
		GPU_f_Cs.sp_buffer = (float *)SVec; 
	}
	__syncthreads(); 

	if(GPU_f_Cs.sv_JacMethod)
		GPU_LMFit(GPU_FitFunction, &GPU_f_Cs, NULL, NULL, GPU_LM_Config, GPU_LMFit_Ins.m, GPU_LMFit_Ins.n, x, 
			NULL, NULL, NULL, NULL,
            InfoNum, Chisq, GPU_LMFit_Ins.GPU_LMFit_Real_Mem,
			GPU_f_Cs.sp_buffer, // The shared buffer in GPU_LMFit starts at sp_smem_left.
			SVecSize, // The total size of shared buffer for GPU_LMFit. 
			bdim*sizeof(float), // The maximum size of shared memory used in user-defined fit function or Jacobian function. 
			GPU_LMFit_Ins.NumOfGPULMFitSharednVec, GPU_LMFit_Ins.NumOfGPULMFitSharedmVec,
			GPU_LMFit_Info); 
	else{
		GPU_LMFit(GPU_FitFunction, &GPU_f_Cs, GPU_AnalyticalJacobian, &GPU_f_Cs, GPU_LM_Config, GPU_LMFit_Ins.m, GPU_LMFit_Ins.n, x, 
			NULL, NULL, NULL, NULL,
            InfoNum, Chisq, GPU_LMFit_Ins.GPU_LMFit_Real_Mem,
			GPU_f_Cs.sp_buffer, // The shared buffer in GPU_LMFit starts at sp_smem_left.
			SVecSize, // The total size of shared buffer for GPU_LMFit. 
			bdim*sizeof(float), // The maximum size of shared memory used in user-defined fit function or Jacobian function. 
			GPU_LMFit_Ins.NumOfGPULMFitSharednVec, GPU_LMFit_Ins.NumOfGPULMFitSharedmVec,
			GPU_LMFit_Info); 
	}
	__syncthreads(); 
}


/* **********************************************************************************************************
		                                                                    CPU host functions
*************************************************************************************************************/
bool Get_GPU_LMFit_Kernel_Config(int n, int m, int userBlkSize, struct cudaDeviceProp deviceProp, 
							struct cudaFuncAttributes KernelAttrib, int *NumOfThreadsPer1DBLK, int *SVecSize, 
							int *NumOfGPULMFitSharednVec, int *NumOfGPULMFitSharedmVec, 
							unsigned long long *GPU_LMFit_Info, char *ErrMsg)
{
	/*
	Function: set up the configuration of GPU for the kernel function - TwoDGauss_GPU_LMFit_Kernel.
	Parameters description:
		n (input) - the number of fitting parameters;
		m (input) - the number of data points;
		userBlkSize (input) - user specified maximum number of threads per CUDA block;
		deviceProp (input) - a struct for CUDA device properties;
		KernelAttrib (input) - a struct for the properties of the kernel function - TwoDGauss_GPU_LMFit_Kernel;
		NumOfThreadsPer1DBLK (output) - the optimized number of threads per CUDA block; 
		SVecSize (output) - the size of the shared memory required to externally allocate for each CUDA block 
			with NumOfThreadsPer1DBLK threads;
		NumOfGPULMFitSharednVec (output) - the number of n-length vectors which can be in shared memory
			in GPU_LMFit;
		NumOfGPULMFitSharedmVec (output) - the number of m-length vectors which can be in shared memory 
			in GPU_LMFit;
		GPU_LMFit_Info (output) - some additional information required by GPU_LMFit;
		ErrMsg (output) - a string pointer for the text of the error description if an error is found.
		return value - true if the user-specified GPU device is initialized successfully, or
						false otherwise. 

	Note:
		All the above three parameters - NumOfGPULMFitSharednVec, NumOfGPULMFitSharedmVec and 
		GPU_LMFit_Info can be easily determined by using the GPU_LMFit accessary function - 
		GPU_LMFit_Num_Of_Shared_Vecs (see also GPU_LMFit_Accessories.h) as shown below. 
	*/

	size_t MaxExtSharedMemPerBlk;
	int MaxThreadsPer1DBlk;

	if(!Get_Kernel_Basic_Config(userBlkSize, deviceProp, KernelAttrib, &MaxThreadsPer1DBlk, 
		&MaxExtSharedMemPerBlk, ErrMsg)) return(false);

	int SMEMAllowedBlkDim; // Max blockDim.x limited the available external shared memory per CUDA block
	SMEMAllowedBlkDim = (int)floor((MaxExtSharedMemPerBlk+0.0f)/
		(GPU_LMFit_Extern_Shared_Mem_Size_Per_Thread+0.0f)); 
	*NumOfThreadsPer1DBLK = MIN((int)nearest_pow2(SMEMAllowedBlkDim), (int)nearest_pow2(m));  
	*NumOfThreadsPer1DBLK = MIN(*NumOfThreadsPer1DBLK, (int)nearest_pow2(MaxThreadsPer1DBlk)); 
	
	/* Minimal SVec size required GPU_LMFit for internal warp reduction. */
	size_t BasicSVecSize = GPU_LMFit_Extern_Shared_Mem_Size_Per_Thread*(*NumOfThreadsPer1DBLK); 
	if(BasicSVecSize>MaxExtSharedMemPerBlk) {
		sprintf_s(ErrMsg, TempStringBufferSize,  
			"%s : not enough shared memory for basic shared SVec!", __FUNCTION__);
		return false;
	}
	
	/* 
	Use the function GPU_LMFit_Num_Of_Shared_Vecs declared in GPU_LMFit_Accessories.h to determine 
	how many n- and m-length vectors can be in shared memory for GPU_LMFit.
	*/
	GPU_LMFit_Num_Of_Shared_Vecs(n, m, (int)(MaxExtSharedMemPerBlk-BasicSVecSize), 
		NumOfGPULMFitSharednVec, NumOfGPULMFitSharedmVec, GPU_LMFit_Info);
	
	// The size of external SVec
	*SVecSize = (int)(BasicSVecSize+(*NumOfGPULMFitSharednVec)*n*sizeof(float)+
		(*NumOfGPULMFitSharedmVec)*m*sizeof(float));

	return true;
}


int GPU_LMFIT_Solver(int n, int m, int NumOfImgs, int ImgDim, int JacMethod, int FitMethod, float * InitialParameters, 
    int DeviceID, float *ImgsBufferPtr, int *userBlkSize, int *userGridSize, float *outx, float *Info, float *Chisq, char *ErrMsg)
{
	/*
	Function: set up the configuration of GPU and implement two-dimensional Gaussian image fittings.
	Parameters description:
		n (input) - the number of fitting parameters;
		m (input) - the number of data points;
		NumOfImgs (input) - the number of images to be fit;
		ImgDim (input) - image dimension size (assume square images);
		JacMethod (input) - if it is 0, a user-defined analytical Jacobian function will be used if it is available,  
			or GPU_LMFit uses its integrated numerical Jacobian function;
		FitMethod (input) - if it is 0, use maximum likelihood estimator in the fit function, or use unweighted 
			least squares;
		init_s (input) - user specified initial value of Gaussian waist width s;
		DeviceID (input) - user-specified GPU device ID number;
		ImgsBufferPtr (input) - the images data in host memory;
		userBlkSize (input) - user specified maximum number of threads per CUDA block;
		userGridSize (input) - user specified maximum number of blocks per CUDA grid;
		outx (output) - fitting parameters in host memory;
		Info (output) - the exit codes of the GPU_LMFit;
		ErrMsg (output) - a string pointer for the text of the error description if an error is found.
		return value - non-zeros if an error is found, or otherwise a zero. 
	*/

	/* CPU Variables */
	memsize_t GPU_Global_Mem_Size;			// The size of total global memory on GPU.
    int GPU_Img_Data_Size, GPU_x_Size, GPU_Info_Size, GPU_Chisq_Size, GPU_LMFit_Real_Mem_Size;
	int ii, CurrImgNum, Total_Num_Of_GPU_LMFit_Kernel_Call, NumOfFits;
	int MaxBLKsPer1DGrid;
	int GPU_LMFit_BLK_Size, GPU_LMFit_SVecSize;
	int NumOfGPULMFitSharednVec, NumOfGPULMFitSharedmVec;
	TWODGAUSS_GPU_LMFIT_INS GPU_LMFit_Paras;
	unsigned long long GPU_LMFit_Info;		// Required by GPU_LMFit and GPU_LMFit_Num_Of_Shared_Vecs
	long int TotalGlobalBufferSize;				// The size of the global memory used on GPU

	// for device
	dim3 dimGPULMFitBlk1D, dimGPULMFitGrid1D;
	struct cudaDeviceProp deviceProp;
	struct cudaFuncAttributes GPULMFitKernel_funcAttrib;

	/* GPU Variables */
	float *GPU_ImgsBuffer;			// Images data buffer in the global memory on GPU.
	float *GPU_x;		// Fitting parameters and background level correction factors in the global memory on GPU.

	int *GPU_info;						// GPU_LMFit exit codes in the global memory on GPU.

	float *GPU_chisq;					// Chi squares in the global memory on GPU.

	float *GPU_LMFit_Real_Mem;	// Single-precision buffer needed by GPU_LMFit in the global memory on GPU.

	/* Initialize cuda device */
	 if(!CuDeviceInit(&deviceProp, DeviceID, ErrMsg)) return(-2);
		
     /* Get kernels' properties */
	if(CheckCudaError(cudaFuncGetAttributes(&GPULMFitKernel_funcAttrib, 
		TwoDGauss_GPU_LMFit_Kernel), ErrMsg, "cudaFuncGetAttributes")) return(-3);

     /* 
     How to determine gridDim.x:
		Although deviceProp.maxGridSize[0] allows 65535 blocks for CUDA Capability 2.0,
		it has been tested and found too many blocks on GPU will cause longer GPU 
		computation time, and windows can automatically restart GPU device after the time 
		defined in Windows (tested in Win7) Regitory Editor -> HKEY_Local_Machine -> 
		System -> CurrentControlSet -> Control -> GraphicsDrivers -> TdrDelay. 
		If GPU is occupied too long by the program and Windows restarts it, 
		then the computation results are wrong, so User parameter userGridSize is important 
		for user to determine a proper gridDim.x for FLIM_GPU_LMFit_Kernel.
	*/
	MaxBLKsPer1DGrid = MIN(NumOfImgs, deviceProp.maxGridSize[0]);
	MaxBLKsPer1DGrid = MIN(MaxBLKsPer1DGrid, *userGridSize); 
	Total_Num_Of_GPU_LMFit_Kernel_Call = (int)ceil((NumOfImgs+0.0f)/MaxBLKsPer1DGrid);	
	
	/* Buffers' sizes */
	GPU_Global_Mem_Size = deviceProp.totalGlobalMem;
	// Parameters' sizes
	GPU_x_Size = NumOfImgs*n*sizeof(float);
	GPU_Img_Data_Size = NumOfImgs*m*sizeof(float);

	if(Info) GPU_Info_Size = NumOfImgs*sizeof(int);
	else GPU_Info_Size = 0;

	if(Chisq) GPU_Chisq_Size = NumOfImgs*sizeof(float);
    else GPU_Chisq_Size = 0;

	/* Set up kernel's configuration */
	//TwoDGauss_GPU_LMFit_Kernel
	if(!Get_GPU_LMFit_Kernel_Config(n, m, 
		*userBlkSize, deviceProp, GPULMFitKernel_funcAttrib, &GPU_LMFit_BLK_Size, 
		&GPU_LMFit_SVecSize, &NumOfGPULMFitSharednVec, &NumOfGPULMFitSharedmVec, 
		&GPU_LMFit_Info, ErrMsg)) 
		return(__LINE__);
	dimGPULMFitBlk1D.x = GPU_LMFit_BLK_Size, dimGPULMFitBlk1D.y = 1, dimGPULMFitBlk1D.z = 1;
	dimGPULMFitGrid1D.x = MaxBLKsPer1DGrid, dimGPULMFitGrid1D.y = 1, dimGPULMFitGrid1D.z = 1;

	/* Allocate GPU memory buffers */
	GPU_LMFit_Real_Mem_Size = MaxBLKsPer1DGrid*(GPU_LMFit_Single_Block_Buffer_Size(n, m) -
		NumOfGPULMFitSharednVec*n*sizeof(float)-NumOfGPULMFitSharedmVec*
		m*sizeof(float));
	TotalGlobalBufferSize = GPU_Img_Data_Size+GPU_x_Size+GPU_Info_Size+ 
							GPU_LMFit_Real_Mem_Size; 

	if((2.0*GPU_Global_Mem_Size/3.0) < TotalGlobalBufferSize){
		// Not support for requiring global mem size > 2/3 of total global mem.
		sprintf_s(ErrMsg, TempStringBufferSize,  
			"%s : Not enough global memory on GPU for required %g MBytes!!!", 
			__FILE__, (TotalGlobalBufferSize+0.0f)/1024.0f/1024.0f);
#ifdef _MY_DEBUG_DISPLAY_
		mexErrMsgTxt(ErrMsg);
#endif
		return(-4);
	} 
	if(CheckCudaError(cudaMalloc((void **)&GPU_x, GPU_x_Size), 
		ErrMsg, "GPU_x : cudaMalloc")) return(__LINE__);	
	if(GPU_Info_Size){
		if(CheckCudaError(cudaMalloc((void **)&GPU_info, GPU_Info_Size), 
			ErrMsg, "GPU_info : cudaMalloc")) return(__LINE__);
	}
	else GPU_info = NULL;
    if (GPU_Chisq_Size){
        if (CheckCudaError(cudaMalloc((void **)&GPU_chisq, GPU_Chisq_Size),
            ErrMsg, "GPU_chisq : cudaMalloc")) return(__LINE__);
    }
    else GPU_chisq = NULL;
	if(CheckCudaError(cudaMalloc((void **)&GPU_ImgsBuffer, GPU_Img_Data_Size), 
		ErrMsg, "GPU_ImgsBuffer : cudaMalloc")) return(__LINE__);
	if(GPU_LMFit_Real_Mem_Size){
		if(CheckCudaError(cudaMalloc((void **)&GPU_LMFit_Real_Mem, GPU_LMFit_Real_Mem_Size), 
			ErrMsg, "GPU_LMFit_Real_Mem : cudaMalloc")) return(__LINE__);
	}
	else GPU_LMFit_Real_Mem = NULL;

	/* Copy the input CPU data to the device global buffer */
	if(CheckCudaError(cudaMemcpy(GPU_ImgsBuffer, ImgsBufferPtr, 
		GPU_Img_Data_Size, cudaMemcpyHostToDevice), 
		ErrMsg, "ImgsBuffer to GPU_ImgsBuffer : cudaMemcpy")) return(__LINE__);

	/* Copy the input CPU initial parameteters to the device global buffer */
	if(CheckCudaError(cudaMemcpy(GPU_ImgsBuffer, ImgsBufferPtr, 
		GPU_Img_Data_Size, cudaMemcpyHostToDevice), 
		ErrMsg, "ImgsBuffer to GPU_ImgsBuffer : cudaMemcpy")) return(__LINE__);
	
	/* Initialize the variables in GPU */
	cudaMemset(GPU_info, 0, GPU_Info_Size);
    cudaMemset(GPU_chisq, 0, GPU_Chisq_Size);
			
	/* Initialize GPU_LMFit_Paras */
	GPU_LMFit_Paras.n = n;
	GPU_LMFit_Paras.m = m;
	GPU_LMFit_Paras.ImgDim = ImgDim;
	GPU_LMFit_Paras.JacMethod = JacMethod;
	GPU_LMFit_Paras.FitMethod = FitMethod;
	GPU_LMFit_Paras.GPU_LMFit_Real_Mem = GPU_LMFit_Real_Mem;
	GPU_LMFit_Paras.NumOfGPULMFitSharednVec = NumOfGPULMFitSharednVec;
	GPU_LMFit_Paras.NumOfGPULMFitSharedmVec = NumOfGPULMFitSharedmVec;

	/* Start ii-for-loop */
	for (ii = 0; ii <Total_Num_Of_GPU_LMFit_Kernel_Call; ii++){
		/* 
		*  Each block completes one fit, so TwoDGauss_GPU_LMFit_Kernel & GPU_2DGau_Init_Kernel 
		*  must have the same gdim. 
		*/
		CurrImgNum = ii*MaxBLKsPer1DGrid;
		NumOfFits = MIN(MaxBLKsPer1DGrid, NumOfImgs-CurrImgNum);
		dimGPULMFitGrid1D.x = NumOfFits;

		if(CheckCudaError(cudaMemset(GPU_LMFit_Real_Mem, 0, GPU_LMFit_Real_Mem_Size), 
			ErrMsg, "zero-initialize GPU_InfoNum : cudaMemset")) return(__LINE__);

		//InitialParameters -> GPU_x
		if(CheckCudaError(cudaMemcpy(GPU_x, InitialParameters, GPU_x_Size, cudaMemcpyHostToDevice), 
		ErrMsg, "InitialParameters to GPU_x : cudaMemcpy"))
            return(__LINE__);
        if (CheckCudaError(cudaDeviceSynchronize(), ErrMsg, "cudaDeviceSynchronize after GPU_2DGau_Init_Kernel"))
            return(__LINE__);

        /* Call kernels */

		TwoDGauss_GPU_LMFit_Kernel<<<dimGPULMFitGrid1D, dimGPULMFitBlk1D, GPU_LMFit_SVecSize>>>(
			GPU_LMFit_Paras, &GPU_x[CurrImgNum*n], &GPU_ImgsBuffer[CurrImgNum*m], 
            &GPU_info[CurrImgNum], &GPU_chisq[CurrImgNum], GPU_LMFit_SVecSize, GPU_LMFit_Info);

		if(CheckCudaError(cudaDeviceSynchronize(), ErrMsg, 
			"cudaDeviceSynchronize after TwoDGauss_GPU_LMFit_Kernel")) 
			return(__LINE__);
	} // End of ii-for-loop

	/* CUDA results return */
	// GPU_x -> outx
	if(CheckCudaError(cudaMemcpy(outx, GPU_x, GPU_x_Size, cudaMemcpyDeviceToHost), 
		ErrMsg, "GPU_x to outx : cudaMemcpy")) return(__LINE__);
	// GPU_info -> CPU_x -> Info
	if(Info){
		if(CheckCudaError(cudaMemcpy(Info, GPU_info, GPU_Info_Size, cudaMemcpyDeviceToHost), 
			ErrMsg, "GPU_info to Info : cudaMemcpy")) return(__LINE__);
		for(ii=0; ii<NumOfImgs; ii++) Info[ii] = (float)(((int *)Info)[ii]);
	}
    if (Chisq)
    {
        if (CheckCudaError(cudaMemcpy(Chisq, GPU_chisq, GPU_Chisq_Size, cudaMemcpyDeviceToHost),
            ErrMsg, "GPU_chisq to Chisq : cudaMemcpy")) return(__LINE__);
    }
	
	/* Free device memory */
	if(GPU_x)
		cudaFree(GPU_x); 
	if(GPU_info)
		cudaFree(GPU_info); 
    if (GPU_chisq)
        cudaFree(GPU_chisq);
	if(GPU_ImgsBuffer)
		cudaFree(GPU_ImgsBuffer); 
	if(GPU_LMFit_Real_Mem)
		cudaFree(GPU_LMFit_Real_Mem);

	/* Normal return */
	return(0); 
}