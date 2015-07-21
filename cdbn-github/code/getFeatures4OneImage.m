%use trained cdbn/crbm models to extract features for a specific image. So
%we have to get patches for this image first,and then use cdbn/crbm model
%to extract features for such patches. 
%params
%spec1,spec2,day
%output
%SVMfeaturefilename: extracted features for each patches (whitendfilename)
%unwhitenfilename: corresponding original unwhitenfilename for these
%patches
%02/24/2015
%by: Dong Nie
function [SVMfeaturefilename,whitenfilename,unwhitenfilename,flag]=getFeatures4OneImage(spec1,spec2,day)
if nargin==0
spec1=2;spec2=2;day=2;
end
%1.generate patch files:data
addpath('../../ourdata/forChris9-19-2014/');
addpath('../../ourdata/forChris9-19-2014/data/');
addpath('../../ourdata/forChris9-19-2014/data/images/2014-04-26 Actino Co-cultures, ISP2, 30 deg, 0.63x1x/');
addpath('../../ourdata/forChris9-19-2014/data/segmentations/2014-04-26 Actino Co-cultures, ISP2, 30 deg, 0.63x1x/');
addpath('../../ourdata/forChris9-19-2014/data/images/2014-04-29 Actino Co-cultures Oatmeal, 30 deg, 0.63x1x/');
addpath('../../ourdata/forChris9-19-2014/data/segmentations/2014-04-29 Actino Co-cultures Oatmeal, 30 deg, 0.63x1x/');
step=4;
[whitenfilename,unwhitenfilename,flag]=testgetGrayPatchesCoverOneImage(spec1,spec2,day,step);%generate patches in patcheswithlabel file
load(['../../patcheswithlabel/',whitenfilename]);
SVMfeaturefilename='';
if flag==0%no such file
    return;
end

%2.extract features by CDBN/crbm model
cdbnpath='../results/cdbnmodel/';
cdbnfile=[cdbnpath,'crbmmodel_traindata100000_20-Feb-2015_layer1.mat'];
load(cdbnfile,'crbm');
model='crbm';
x=trainFeatureMat;
trainedlabel=trainLabelMat;


if model=='crbm'
    graypatchwidth=sqrt(size(x,1));%the initial input x is vetor form, here i input rgb patch
    rgbpatchwidth=sqrt(size(x,1)/3);%rgb
    if graypatchwidth==floor(graypatchwidth)%it is an gray image
        x=reshape(x,[graypatchwidth,graypatchwidth,1,size(x,2)]);% the third parameter is channel number
        disp('input is gray images\n');
    elseif rgbpatchwidth==floor(rgbpatchwidth)%it is a rgb   
        x=reshape(x,[rgbpatchwidth,rgbpatchwidth,3,size(x,2)]);% the third parameter is channel number
        disp('input is rgb images\n');
    else
        disp('input error\n');
    end
    pars=paramsetup();
    pars.std_gaussian=pars.sigma_stop;
    [layer1features,layer1pooledfeatures,layer1states]=getFeaturesByCRBMmodel(x,crbm,pars);
    layer1pooledstates=maxPooling(layer1states);
    svmfeaturepath='../results/feature4SVM/maskOneImage/';
    SVMfeaturefilename=sprintf('features4SVM4spec%d_%d_day%d_layer1.mat',spec1,spec2,day);
    save([svmfeaturepath,SVMfeaturefilename],'layer1features','layer1pooledfeatures','layer1pooledstates','trainLabelMat','specday','rows','cols');%when use combine feature
else
    [layer1features,layer1pooledfeatures,layer1pooledstates,layer2features,layer2pooledfeatures,layer2pooledstates]=getFeaturesByCDBNmodel(x,cdbn,pars);%This is fast, it takes about 9 minutes to extract features for 5,000 patches.
    svmfeaturepath='../results/feature4SVM/maskOneImage/';
    SVMfeaturefilename=sprintf('features4SVM4spec%d_%d_day%d_4layers',spec1,spec2,day);
    save([svmfeaturepath,SVMfeaturefilename],'layer1features','layer1pooledfeatures','layer1pooledstates','layer2features','layer2pooledfeatures','layer2pooledstates','trainLabelMat','specday','rows','cols');%when use combine features, we can directly combine two matrices
end
end