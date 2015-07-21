%This script is written to 
%1.take all patches from an image you did not train on
%2.get features for each patch using cdbn model and process to format
%fitting svm model
%3.predict background and forground patches using svm model
%4.mask out background patches, so if a patch is predicted to be background
%just turn all of its pixels black in the original image.This will be an image with masked background.
%5. produce an image with masked foreground 
function workExample2()
 fpath='../bacteria/';
 filename=[fpath,'1-1 2d.jpg'];
%[patchset,precomp]=getPatchesCoverOneImage(filename,16,16);
% patchfilename='../results/maskImage/data-coverOneImage28-Oct-2014.mat';
% load(patchfilename,'trainFeatureMat');
% patchset=trainFeatureMat;
% 
% testsvmdatapath=extractfeatures(patchset);
%fpreLablePath=preLabelbySVM(testsvmdatapath);
%maskImages(fpreLablePath);
maskImages();
return

function testsvmdatapath=extractfeatures(patchset)
cdbnpath='..\results\cdbnmodel\cdbnmodel_layer2inputpooled\';
cdbnfile=[cdbnpath,'crbm_new1h_traindata_V1_w10_b36_p0.5_pl0.01_plambda2_sp2_CD_eps0.001_l2reg0.5_bs100_20141012T005543_model_1000.mat'];
pars=paramsetup();
cdbn=cdbnsetup();
load(cdbnfile,'cdbn');%I have train a cdbn model%only use 1000
testdata=patchset;
testfeatures=getFeaturesByCDBNmodel(testdata,cdbn,pars);
num=length(testfeatures);
testlabel=zeros(num,1);%make it all to 0, and it is not all the true label
testlabel=squeeze(testlabel);

testsvmdatapath=processfeaturevectors2SVM(testfeatures,testlabel);
return

%use svm model to predict labels
function fpreLablePath=preLabelbySVM(testsvmdatapath)
svmmodelpath='..\results\feature4SVMlayer2inputpooled_cdbnmodel1000\feature4SVM5000instance\bernoullifeatures\';
layer1svmmodel=[svmmodelpath,'layer1feature4svm_train.dat.model'];
layer2svmmodel=[svmmodelpath,'layer2feature4svm_train.dat.model'];
layer3svmmodel=[svmmodelpath,'layer3feature4svm_train.dat.model'];
layer4svmmodel=[svmmodelpath,'layer4feature4svm_train.dat.model'];
%predict foreground and background
%I use binary programs to do the classification work.
%The predicted labels are in '..\results\maskImage\'
return

%to make maskImages
function maskImages(fpreLablePath)
if nargin==0
    fPreLabelPath='..\results\maskImage\feature4 1-1-2 d databycdbn1000\colwisepatches_givencode\';
end
preLabelFile1=[fPreLabelPath,'combine_layer13_result_trim'];
preLabels1=load(preLabelFile1);
preLabels1=squeeze(preLabels1);
label=preLabels1;
ratioed=0;%原图是否调了比例
[numofrows,numofcols]=getRowCol(ratioed);


patchfile=[fPreLabelPath,'data-coverOneImage29-Oct-2014 unwhiten.mat'];
load(patchfile,'trainFeatureMat','rows','cols','labels');
%load(patchfile,'patchset');
numofrows=rows;
numofcols=cols;
%label=labels;


label=trimLabel(label,numofrows,numofcols);

patchset=trainFeatureMat;

originalImage=restoreImage(patchset,numofrows,numofcols);
originalImage=uint8(originalImage);

%first mask background patches
dim=size(patchset,1);
temp=patchset;
for i=1:length(label)
    if label(i)==-1%background
        temp(:,i)=0;%0->black
    end
end
%show temp
maskBackgroundImage=restoreImage(temp,numofrows,numofcols);
maskBackgroundImage=uint8(maskBackgroundImage);

%then mask the foreground
temp2=patchset;
for i=1:length(label)
    if label(i)==1%foreground
        temp2(:,i)=0;%0->black
    end
end
maskForegroundImage=restoreImage(temp2,numofrows,numofcols);
maskForegroundImage=uint8(maskForegroundImage);
%show temp2

figure(1);
imshow(originalImage);
saveas(gcf, sprintf('%soriginal.png',fPreLabelPath));
figure(2);
imshow(maskBackgroundImage);
saveas(gcf, sprintf('%smaskBackground.png',fPreLabelPath));
figure(3);
imshow(maskForegroundImage);
saveas(gcf, sprintf('%smaskForeground.png',fPreLabelPath));
return

function J=restoreImage(temp,numofrows,numofcols,patchrow,patchcol)
if nargin<5
    patchrow=16;
    patchcol=16;
end

%[numofrows,numofcols]=getRowCol();

for i=1:numofrows
    for j=1:numofcols
        vector=temp(:,(i-1)*numofcols+j);
        if length(vector)==patchrow*patchcol
            patch=reshape(vector,[patchrow,patchcol]);
        else
            patch=reshape(vector,[patchrow,patchcol,3]);
            patch=rgb2gray(patch);
        end
        J(((i-1)*16+1):i*16,((j-1)*16+1):j*16)=patch;
    end
end
return

function [numofrows,numofcols]=getRowCol(ratioed,patchrow,patchcol)
if nargin==1
    patchrow=16;
    patchcol=16;
end
fpath='../bacteria/';
filename=[fpath,'1-1 2d.jpg'];
I=imread(filename);
I=rgb2gray(I);
if ratioed
    ratio = min([512/size(I,1), 512/size(I,2), 1]);
    if ratio<1
        I = imresize(I, [round(ratio*size(I,1)), round(ratio*size(I,2))], 'bicubic');
    end
end
[rows,cols]=size(I);
numofrows=floor(rows/patchrow);
numofcols=floor(cols/patchcol);
return