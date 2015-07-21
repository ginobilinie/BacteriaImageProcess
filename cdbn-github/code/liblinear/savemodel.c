#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <ctype.h>  
#include "svm.h"  
  
#include "mex.h"  
#include "svm_model_matlab.h"  
  
  
void exit_with_help()  
{  
    mexPrintf(  
    "Usage: savemodel('filename', model);\n"  
    );  
}  
  
int savemodel(const char *filename, const mxArray *matlab_struct)  
{  
    const char *error_msg;  
    struct svm_model* model;  
    int result;  
    model = matlab_matrix_to_model(matlab_struct, &error_msg);  
      
    if (model == NULL)  
    {  
        mexPrintf("Error: can't read model: %s\n", error_msg);  
    }  
      
    result = svm_save_model(filename, model);  
      
    if( result != 0 )  
    {  
        mexPrintf("Error: can't write model to file!\n");  
    }  
      
    return result;  
}  
  
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  
{  
    if(nrhs == 2)  
    {  
        char filename[256];  
        int *result;  
          
        mxGetString(prhs[0], filename, mxGetN(prhs[0])+1);  
          
        plhs[0] = mxCreateNumericMatrix(1, 1, mxINT8_CLASS, 0);  
          
        result = mxGetPr(plhs[0]);  
        *result = savemodel(filename, prhs[1]);  
    }  
    else  
    {  
        exit_with_help();         
        return;  
    }  
} 