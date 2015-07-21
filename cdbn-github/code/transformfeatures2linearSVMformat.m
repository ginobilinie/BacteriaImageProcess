%This script is written to transform features to fit for linear svm form:
%matrix form.
%here, note, it is different from processfeaturevectors2SVM.m which
%transform features and labels to: label fid:fvalue...
%this script just turn cell form features to matrix form
%params:
%features:a cell form:each cell unit contain 4 layer binary feature
%the structure of cell: 
%1.numofpatches cells (each cell represent for a patch)
%2.for each cell, there are two cells inside to represent two convolution
%layers
%3.poshidstates, poshidprobs,pooledfeatures
%output:
%layer1Matrix:dimoffeatures*numofpatches
%layer2...
%layer3Matrix:dimoffeatures*numofpatches
%layer4...
function [mat1,mat2,mat3,mat4]=transformfeatures2linearSVMformat(features)
numofpatches=length(features);
dim1=length(features{1}{1}.poshidstates);
dim2=length(features{1}{1}.pooledstates);
dim3=length(features{1}{2}.poshidstates);
dim4=length(features{1}{2}.pooledstates);
mat1=zeros(dim1,numofpatches);
mat2=zeros(dim2,numofpatches);
mat3=zeros(dim3,numofpatches);
mat4=zeros(dim4,numofpatches);
for i=1:numofpatches
    patch=features{i};
    layer1=patch{1};
    tt=layer1.poshidstates;
    mat1(:,i)=layer1.poshidstates;
    mat2(:,i)=layer1.pooledstates;
    layer2=patch{2};
    mat3(:,i)=layer2.poshidstates;
    mat4(:,i)=layer2.pooledstates;
end
%mat3=[mat1;mat2];%按列合并,得到合并的features,but when the matrix is too large, it
%get out of memory error
end
