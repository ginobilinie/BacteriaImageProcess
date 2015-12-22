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
    "Usage: model = loadmodel('filename', num_of_feature);\n"  
    );  
}  
  
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  
{  
    if(nrhs == 2)  
    {  
        char filename[256];  
        int num_of_feature;  
        struct svm_model* model;  
        int featurenum;  
        const char *error_msg;  
          
        mxGetString(prhs[0], filename, mxGetN(prhs[0])+1);  
        model = svm_load_model(filename);  
          
        if(model == NULL)  
        {  
            mexPrintf("Error: can't read the model file!\n");  
            return;  
        }  
          
        featurenum = *(mxGetPr(prhs[1]));  
          
        error_msg = model_to_matlab_structure(plhs, featurenum, model);  
          
        if(error_msg)  
            mexPrintf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);  
          
        svm_free_and_destroy_model(&model);  
    }  
    else  
    {  
        exit_with_help();         
        return;  
    }  
}