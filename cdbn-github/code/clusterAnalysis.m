%This script is to do cluster analysis for one image represented by
%features extracted from cdbn model.
function [T1,T2,T3,T4]=clusterAnalysis()
fpath='../results/feature4SVM/maskOneImage/';
%featurefile=[fpath,'features4SVMcoverOneImage_4layers.mat'];

featurefile=[fpath,'features4firstImage_4layers.mat'];
load(featurefile);%layer1featuremat, layer2...layer4
% Y = pdist(X, 'euclid');
% Z = linkage(Y, 'single');
% T = cluster(Z, 'cutoff', CUTOFF);
CUTOFF=20;
% Y1 = pdist(layer1featuremat', 'euclid');
% Z1 = linkage(Y1, 'single');
% T1 = cluster(Z1, 'maxclust', CUTOFF);
% 
% Y2 = pdist(layer2featuremat', 'euclid');
% Z2 = linkage(Y2, 'single');
% T2 = cluster(Z2, 'maxclust', CUTOFF);

Y3 = pdist(layer3featuremat', 'euclid');
Z3 = linkage(Y3, 'single');
T3 = cluster(Z3, 'maxclust', CUTOFF);
% T1=[];T2=[];T3=[];T4=[];
Y4 = pdist(layer4featuremat', 'euclid');
Z4 = linkage(Y4, 'single');
T4 = cluster(Z4, 'maxclust', CUTOFF);
save('../results/clusterAnalysis/clusterCoverOneImage.mat','T3','T4');
figure(1),hist(T3);
figure(2),hist(T4);

return


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


