%This script is written to view pooled states projected in image space.
%There are totally 4 main functions in this script
%1.generate patches files: patches for one image or for several patches,
%edged, whitened or ..
%2.extract features by CDBN/crbm model for the generated patches
%3.visualize the pooled features, and store the fired patches for a
%specific filter (out of 35 filters).
%4.project the 2nd/1st layer (pooled) features down into original image space, which we can think as reconstruction (or deconvolution process)  
%2/11/2015
%by Dong Nie
function y=viewPooledFeature()
inputpath='../results/feature4SVM/maskOneImage/';
outputpath='../results/cdbnvisual/specday/';

spec1=10;spec2=8;day=2;
process3=0;
process4=1;
%1.generate patch files:data
addpath('../../ourdata/forChris9-19-2014/');
addpath('../../ourdata/forChris9-19-2014/data/');
addpath('../../ourdata/forChris9-19-2014/data/images/2014-04-26 Actino Co-cultures, ISP2, 30 deg, 0.63x1x/');
addpath('../../ourdata/forChris9-19-2014/data/segmentations/2014-04-26 Actino Co-cultures, ISP2, 30 deg, 0.63x1x/');
whitenfilepath=testgetEdgePatchesCoverOneImage(spec1,spec2,day);%generate patches in patcheswithlabel file
load(['../../patcheswithlabel/',whitenfilepath]);

%2.extract features by CDBN model
cdbnpath='../results/cdbnmodel/';
cdbnfile=[cdbnpath,'cdbn_8428_p0.002__plambda10_sp2_CD_eps0.001_l2reg0.0001_bs100_20150208T040625.mat_model_8428.mat'];
load(cdbnfile,'cdbn');
traindata=trainFeatureMat;
trainedlabel=trainLabelMat;
[layer1features,layer1pooledfeatures,layer1pooledstates,layer2features,layer2pooledfeatures,layer2pooledstates]=getFeaturesByCDBNmodel(traindata,cdbn);%This is fast, it takes about 9 minutes to extract features for 5,000 patches.
svmfeaturepath='../results/feature4SVM/maskOneImage/';
%spec1=1;spec2=8;day=2;
save([svmfeaturepath,sprintf('features4SVMcoverOneImage_4layers_spec%d%d_%dd.mat',spec1,spec2,day)],'layer1features','layer1pooledfeatures','layer1pooledstates','layer2features','layer2pooledfeatures','layer2pooledstates','trainedlabel','specday','coords','rows','cols');%when use combine features, we can directly combine two matrices

%3.visualize pooled states of 2nd layer
if process3==1
datapath=[inputpath,sprintf('features4SVMcoverOneImage_4layers_spec%d%d_%dd.mat',spec1,spec2,day)];
load(datapath,'layer2pooledstates','layer1pooledstates','layer2features','trainedlabel','coords');
originaldata=sprintf('../../patcheswithlabel/data-coverColorImageSpec%d_%d_%dd12-Feb-2015 unwhiten_112112_sub1_step56_isfore1_isback1.mat',spec1,spec2,day);
ps=layer2pooledstates;
ps=reshape(ps,[size(ps,1)*size(ps,2),size(ps,3),size(ps,4)]);
ind=[1,3,10];
%ind=ind+100;
ind=find(trainedlabel==1);
load(originaldata,'trainFeatureMat');
width=sqrt(size(trainFeatureMat,1)/3);
tt=reshape(trainFeatureMat,[width,width,3,size(trainFeatureMat,2)]);

%view patches which reflect to specific group
map=zeros(size(ps,2),length(ind));
for k=1:size(ps,2)%bases
    for i=1:length(ind)
        display_network_layer1(ps(:,:,ind(i)));
        fki=ps(:,k,ind(i));%k'th group for image i
        if sum(fki(:))>0
            map(k,i)=1;
        end
        %saveas(gcf,[outputpath,sprintf('%sps-spec%d%d-%dd36channels4patch%d.png',date,spec1,spec2,day,ind(i))]);
        %imwrite(tt(:,:,:,ind(i)),[outputpath,sprintf('%sps-spec%d%d-%ddoriginalImage4patch%d.png',date,spec1,spec2,day,ind(i))]);
    end
end
%choose relative patches for each fired filter
for k=1:size(ps,2)
    fk=ind(find(map(k,:)==1));
    [s,mess,messid] = mkdir([outputpath,sprintf('species%d_%d/',spec1,spec2)],sprintf('filter%d',k));
    if (~s)
        fprintf('create dir failed');
    end
    for i=1:length(fk)
        imwrite(tt(:,:,:,fk(i)),[[outputpath,sprintf('species%d_%d/',spec1,spec2)],sprintf('filter%d/%sps-spec%d%d-%ddoriginalImage4patch%d.png',k,date,spec1,spec2,day,fk(i))]);
        %write edges for this patch
        J=tt(:,:,:,fk(i));
        J=rgb2gray(J);
        J=edge(J,'canny');
        imwrite(J,[[outputpath,sprintf('species%d_%d/',spec1,spec2)],sprintf('filter%d/%sps-spec%d%d-%dd_edgeImage4patch%d.png',k,date,spec1,spec2,day,fk(i))]);
    end
end
%view pooled features
for i=1:length(ind)
    display_network_layer1(ps(:,:,ind(i)));
    saveas(gcf,[outputpath,sprintf('%sps-spec%d%d-%dd36channels4patch%d.png',date,spec1,spec2,day,ind(i))]);
    imwrite(tt(:,:,:,ind(i)),[outputpath,sprintf('%sps-spec%d%d-%ddoriginalImage4patch%d.png',date,spec1,spec2,day,ind(i))]);
end

end

%4.project pooled layers to image space: reconstruct visible layer
if process4==1
layer=1;
recondata=projectPoolLayers2ImageSpace(layer1features,cdbn,1);
elseif layer==2
r=reshape(recondata,625,391);
end
if layer==1
    r=reshape(recondata,27*27,391);
ind=find(trainedlabel==1);
display_network_layer1(r(:,ind));
saveas(gcf,sprintf('../results/cdbnvisual/specday/spec%d%d_%dd_recon.png',spec1,spec2,day));
end
end