%This script is written to project pooled features/states into image space,
%which means I trace back to the orginal image space, go down from the
%pooled layers to visible layer, by using reconstruct function
%params:
%layer: 1 or 2
%pool or not
%pooled or conv features
%cdbn model

%Date:2/12/2014
%by: Dong Nie
function [reconprobsFV]=projectPoolLayers2ImageSpace(layerfeatures,cdbn,layer)
if nargin==0
    layer=2;
    featurefile='../results/feature4SVM/maskOneImage/features4SVMcoverOneImage_4layers.mat';
    load(featurefile);%here I just took layer 2 features instead of layer2 pooled features/states
    cdbnfile='../results/cdbnmodel/cdbn_8428_p0.002__plambda10_sp2_CD_eps0.001_l2reg0.0001_bs100_20150208T040625.mat_model_8428.mat';
    load(cdbnfile,'cdbn');
end



%layer2 output-> layer2 input
if layer==2
layer2features=layerfeatures;
width=4;
channel=36;
layer2features=reshape(layer2features,[width,width,channel,size(layer2features,2)]);
x=layer2features;
pars=paramsetup(2);
pars.std_gaussian=pars.sigma_stop;
crbm=cdbn.crbm{1,2};
[reconprobsFV]=projectLayersDownByCRBMModel(x,crbm,pars);

%layer2 input -> layer1 output
layer1features=unPooling(reconprobsFV);
end
%layer1 output->visible layer
if layer==1
    layer1features=layerfeatures;
    channel=36;
    width=20;
    width=20;
    layer1features=reshape(layer1features,[width,width,channel,size(layer1features,2)]);
end;
crbm=cdbn.crbm{1,1};
pars=paramsetup(1);
pars.std_gaussian=pars.sigma_stop;
[reconprobsFV]=projectLayersDownByCRBMModel(layer1features,crbm,pars);

return

%This function is to reconstruct a lower layer
function [reconprobsFV]=projectLayersDownByCRBMModel(x,crbm,pars)
numofpatches=size(x,4);
batchsize=pars.batchsize;

numofbatches=ceil(numofpatches/batchsize);

for i=1:numofbatches
    batchindex=(i-1)*pars.batchsize+1:min(i*batchsize,numofpatches);
      imdata_batch=x(:,:,:,batchindex);
    [recon]=getlowerlayer(imdata_batch,crbm,pars);%should use poshidprobs
    reconprobsFV(:,:,:,batchindex)=recon;
end
return

%now I have to solve unpooling problem: in simple way, just reflect one
%pixel to 4 pixels, one with the max value, the three others are 0. The key
%problem is the location of the max value in 4 positions, to simplify,
%choose a random position, to make it sure, when pooling, just remember the
%the position for this value
%params:
%hinput:[width,width,channel,num]
%output:
%loutput:[2width,2width,channel,num]
function loutput=unPooling(hinput)
width=size(hinput,1);
channel=size(hinput,3);
num=size(hinput,4);
loutput=zeros(2*width,2*width,channel,num);
for i=1:width
    for j=1:width
        %loutput(2*i-1,2*j-1,:,:)=hinput(i,j,:,:);
        loutput(2*i-1,2*j-1,:,:)=1/1*hinput(i,j,:,:);
        %loutput(2*i-1,2*j,:,:)=1/1*hinput(i,j,:,:);
        %loutput(2*i,2*j-1,:,:)=1/1*hinput(i,j,:,:);
        %loutput(2*i,2*j,:,:)=1/1*hinput(i,j,:,:);
    end
end
return

function [recon]=getlowerlayer(x,crbm,pars)
    imdata=x;%
    %imdata = trim_image_for_spacing_fixconv(imdata, crbm.filtersize, pars.spacing);%trim
    W=crbm.W;
    % do convolution/ get poshidprobs
    %if pars.currentLayer==1
    recon = crbm_reconstruct_gaussian(x, W, pars);%recon data
     
return

function im2 = trim_image_for_spacing_fixconv(im2, filtersize, spacing)
% % Trim image so that it matches the spacing.
if mod(size(im2,1)-filtersize+1, spacing)~=0
    n = mod(size(im2,1)-filtersize+1, spacing);
    im2(1:floor(n/2), : ,:,:) = [];
    im2(end-ceil(n/2)+1:end, : ,:,:) = [];
end
if mod(size(im2,2)-filtersize+1, spacing)~=0
    n = mod(size(im2,2)-filtersize+1, spacing);
    im2(:, 1:floor(n/2), :,:) = [];
    im2(:, end-ceil(n/2)+1:end, :,:) = [];
end
return