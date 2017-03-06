/* Main program */

#include "Apps.cuh"
#include "D:\Program Files (x86)\MATLAB\R2014b\extern\include\mex.h"
#include "GPU_2DGauss_Solver.cuh" // To use GPU_LMFIT_Solver

/* GPU CUDA Libraries */
#include "cuda_runtime_api.h"

/* The Matlab mex interface function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Input and basic parameters */
	int n = 5;					// Number of fitting parameters
	int m;						// Number of data points 
	int NumOfImgs;			// Number of images
	int ImgDim;				// Image dimension size (assume square images)
	int FitMethod;			// 0 - maximum likelihood estimator, or 1 - unweighted least squares. 
	int JacMethod;			// 0 - Analytical Jacobian; 1 - Numerical Jacobian

	// Images data buffer
	float *ImgsBuffer;			// The pointer to images data buffer
	unsigned int ImgDataSize;	// Images data buffer size
	float *InitialParameters;			// The pointer to initial parameters
	
	// GPU CUDA
	int GPU_device_ID;		// GPU device ID
	int GPU_Block_Size;		// User-defined maximum blockDim.x (blockDim.y = blockDim.z = 1)
	int GPU_Grid_Size;		// User-defined maximum gridDim.x (gridDim.y = gridDim.z = 1)

	/* Output */
	float *outx;				// Fitted parameters
	char UsageMsg[TempStringBufferSize] = {};		// A string for the information to display the function usage.
	char ErrorMsg[TempStringBufferSize+1] = {};	// A string for the information of errors from the function 
	float *Info = NULL;		// To return either Chi squares or infomation code from GPU-LMFit
    float *Chisq = NULL;
	/* Prepare the usage string */
	//sprintf_s(UsageMsg, TempStringBufferSize,  
	//	"\n\t%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n"
	//	"%s\n%s\n%s\n%s\n%s\n%s\n\n",
	//	"Useage: x = GPU2DGaussFit(ImgsData, ImgDimension, Init_s, FitMethod, JacMethod, ...",
	//	"                     GPU_device_ID, GPU_Block_Size, GPU_Grid_Size).",
	//	"        or [x infonum] = GPU2DGaussFit(ImgsData, ImgDimension, Init_s, FitMethod, ...", 
	//	"                     JacMethod, GPU_device_ID, GPU_Block_Size, GPU_Grid_Size).",
	//	"        ImgsData is 1D square image data array (single data type).",
	//	"              If ImgsData is originally 2D, it need to be converted to 1D.",
	//	"        ImgDimension is the dimension size of the images. It must be a scalar.",
	//	"        Init_s is the initial value of the Gaussian waist width. (default = 1.0f pixel)",
	//	"        FitMethod is to select fitting methods (estimators): ",
	//	"              0 - maximum likelihood estimator (MLE);",
	//	"              1 - unweighted least squares (uWLS);",
	//	"        JacMethod = 0 is to use analytical Jacobian matrix, or otherwise numerical", 
	//	"              Jacobian matrix. (Note: no analytical Jacobian function is available for MLE).",
	//	"        GPU_device_ID is the device number of GPU to be used", 
	//	"             (it is zero if only one GPU is available);",
	//	"        GPU_Block_Size and GPU_Grid_Size are for user to preset the maximum block ",
	//	"              size and the maximum grid size of CUDA, respectively;",
	//	"        x is fitted parameters, infonum is the status of each fit.");

	/* Check for proper number of arguments */
	if(nrhs!=8) {
		mexPrintf("%s", UsageMsg);
		mexErrMsgTxt("Eight inputs required!");
	}
    if (nlhs != 1 && nlhs != 2 && nlhs != 3) {
		mexPrintf("%s", UsageMsg);
		mexErrMsgTxt("Incorrect output parameters!");
	}

	/* Check the formats for the input arguments */
	if
		( 
		!mxIsSingle(prhs[0]) || mxIsComplex(prhs[0]) ||
		mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1])!=1 || 
		mxIsComplex(prhs[2]) || mxIsComplex(prhs[2]) ||
		mxIsComplex(prhs[3]) || mxGetNumberOfElements(prhs[3])!=1 ||
		mxIsComplex(prhs[4]) || mxGetNumberOfElements(prhs[4])!=1 ||
		mxIsComplex(prhs[5]) || mxGetNumberOfElements(prhs[5])!=1 ||
		mxIsComplex(prhs[6]) || mxGetNumberOfElements(prhs[6])!=1 ||
		mxIsComplex(prhs[7]) || mxGetNumberOfElements(prhs[7])!=1
		)
	{
		mexPrintf("%s", UsageMsg);
		mexErrMsgTxt("Input parameters are wrong!");
	}
	
	/* Check the format of input image data */
	if(mxGetM(prhs[0])!=1)
		mexErrMsgTxt("ImgsData must be a row vector.");

	/* Get the image dimension size */
	ImgDim = (int)mxGetScalar(prhs[1]);

	/* m - number of pixels */
	m = ImgDim*ImgDim;
	
	/* Get the size of the input image data */
	ImgDataSize = (unsigned int)mxGetN(prhs[0]);

	/* Calculate the number of images*/
	NumOfImgs = (int)(ImgDataSize/ImgDim/ImgDim);

	/* Check the input image data validation */
	if ((NumOfImgs-(float)ImgDataSize/ImgDim/ImgDim) == 0.0) {
		
		/* Get and check the fitting method */
		FitMethod = (int)mxGetScalar(prhs[3]);
		if (FitMethod<0 || FitMethod>1) {
			mexPrintf("%s", UsageMsg);
			mexErrMsgTxt("Wrong Fit methods.");
		}

		/* Get Jacobian method */
		JacMethod = (int)mxGetScalar(prhs[4]);
		if(JacMethod!=0) JacMethod = 1;

		/* Create a pointer to the image data */
		ImgsBuffer = (float *)mxGetPr(prhs[0]);

		InitialParameters = (float *)mxGetPr(prhs[2]);

		/* Get the user-defined maxiumum CUDA block and grid size*/
		GPU_Block_Size = (int)mxGetScalar(prhs[6]);
		GPU_Grid_Size = (int)mxGetScalar(prhs[7]);

		/* Prepare the output matrices */
		// Fitted x
		plhs[0] = mxCreateNumericMatrix(1, n*NumOfImgs, mxSINGLE_CLASS, mxREAL);
		outx = (float *) mxGetData(plhs[0]);
		if(outx == NULL)
			mexErrMsgTxt("Fail to allocate memory for output x!");
		// Info
        if (nlhs == 2 || nlhs == 3) {
			plhs[1] = mxCreateNumericMatrix(1, NumOfImgs,  mxSINGLE_CLASS, mxREAL);
			Info = (float *) mxGetData(plhs[1]);
			if(Info == NULL)
				mexErrMsgTxt("Fail to allocate memory for output Info!");
		}
        if (nlhs == 3) {
            plhs[2] = mxCreateNumericMatrix(1, NumOfImgs, mxSINGLE_CLASS, mxREAL);
            Chisq = (float *)mxGetData(plhs[2]);
            if (Chisq == NULL)
                mexErrMsgTxt("Fail to allocate memory for output Chisq!");
        }
		
		/* Get the user specified GPU device ID */
		GPU_device_ID = (int)mxGetScalar(prhs[5]);

		if(GPU_device_ID == -1){
			/* 
			One may add MPFit routine here for CPU version fitting: MPFit is originally developed 
			based on Minpack by Craig Markwardt. For more details, please refer to 
			http://cow.physics.wisc.edu/~craigm/.
			*/
			mexErrMsgTxt("GPU computation only! No CPU version fitting routine!");
		}
		else{
			if(GPU_LMFIT_Solver(n, m, NumOfImgs, ImgDim, JacMethod, FitMethod, InitialParameters, 
				GPU_device_ID, ImgsBuffer, &GPU_Block_Size, &GPU_Grid_Size, outx, Info, Chisq, ErrorMsg))
				mexErrMsgTxt(ErrorMsg);
		}
	}
	else
		mexErrMsgTxt("Image Data Size, Dimension and Number of Images do NOT match!");
}
