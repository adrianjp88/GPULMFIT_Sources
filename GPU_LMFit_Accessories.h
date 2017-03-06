#ifndef _GPU_LMFIT_ACCESSORIES_H_
#define _GPU_LMFIT_ACCESSORIES_H_

/*
	Prototype: 
		extern int GPU_LMFit_Single_Block_Buffer_Size(int n, int m);
	Function: 
		To determine the global buffer size of each block of GPU_LMFit threads.
		Here the input paramters n and m must be the same as those in GPU_LMFit.
		Please see also the discription for the input parameter GPU_Real_Mem of 
		GPM_LMFit in GPU_LMFit.cuh. 

*/
extern int GPU_LMFit_Single_Block_Buffer_Size(int, int);

/* 
	Prototype: 
		extern void GPU_LMFit_Num_Of_Shared_Vecs(int n, int m, int SMEM_Size, 
							int *NumOfSharednVec, int *NumOfSharedmVec, 
							unsigned long long *Info);
	Function: 
		To determine the numbers *NumOfSharednVec and *NumOfSharedmVec, which 
		are, respectively, the numbers of n- and m-element vectors which have been 
		allocated external shared memory. This function also return a 64 bit parameter
		*Info including the information about the version and the license of GPU_LMFit.

	Inputs:
		n and m must be the same as those in GPU_LMFit;
		SMEM_Size is the maximum extern shared memory per block, which should be 
			calculated by subtracting the size of the static shared memory (can be known 
			by calling the CUDA function cudaFuncGetAttributes) in a kernel function from 
			the maximum size of shared memory per block. 

	Outputs:
		NumOfSharednVec, NumOfSharedmVec and Info.
*/
extern void GPU_LMFit_Num_Of_Shared_Vecs(int, int, int, int *, int *, unsigned long long *);

#endif //  _GPU_LMFIT_ACCESSORIES_H_
