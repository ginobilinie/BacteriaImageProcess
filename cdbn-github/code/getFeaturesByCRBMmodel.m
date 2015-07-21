%This script is written to get features given a crbm model(network
%parameters), and form the features for one object to a vector 
%crbm:crbm model 
%input:x is the patch in form of [row,col,channel,num]
function [poshidprobsFv,pooledprobsFV,poshidstatesFV]=getFeaturesByCRBMmodel(x,crbm,pars)
%load(fpath,'cdbn');
numofpatches=size(x,4);
batchsize=pars.batchsize;

numofbatches=ceil(numofpatches/batchsize);

for i=1:numofbatches
    batchindex=(i-1)*pars.batchsize+1:min(i*batchsize,numofpatches);
      imdata_batch=x(:,:,:,batchindex);

%     if (length(xi)==inputrow*inputcol)%if it is gray, we donot need to transfer to gray
%         xi=reshape(xi,[inputrow,inputcol]);
%     else
%         xi=reshape(xi,[inputrow,inputcol,3]);%if it is rgb, we need to transfer to gray
%         xi=rgb2gray(xi);
%     end
    [poshidprobs,pooledprobs,poshidstates]=getfeature(imdata_batch,crbm,pars);%should use poshidprobs
    poshidstatesFV(:,:,:,batchindex)=poshidstates;
    pooledprobsFV(:,:,:,batchindex)=pooledprobs;
    poshidprobsFv(:,:,:,batchindex)=poshidprobs;
end

return

function [poshidprobs,pooledprobs,poshidstates]=getfeature(x,crbm,pars)
    imdata=x;%
    imdata = trim_image_for_spacing_fixconv(imdata, crbm.filtersize, pars.spacing);%trim
    W=crbm.W;
    hbias_vec=crbm.hbias_vec;
    vbias_vec=crbm.vbias_vec;
    % do convolution/ get poshidprobs
    %if pars.currentLayer==1
        poshidexp = crbm_inference_softmax(imdata, W, hbias_vec, pars);%here get: I(hij)=hbias+W*V
        [poshidstates poshidprobs] = crbm_sample_multrand2(poshidexp, pars.spacing);%softmax(I(hij)) to get P(h=1|v) binary state 
        pooledprobs=probabilisticMaxPooling(poshidprobs);
%     else
%         poshidprobs = crbm_inference_sigmoid(imdata, W, hbias_vec, pars);
%         poshidstates = double(poshidprobs > rand(size(poshidprobs))); 
%         pooledprobs=maxPooling(poshidprobs);
%     end
    %pooledprobs=maxPooling(poshidprobs);
    %poshidstates=squeeze(reshape(poshidstates,[size(poshidstates,1)*size(poshidstates,2)*size(poshidstates,3),1])); 
    %pooledstates=squeeze(reshape(pooledfeatures,[size(pooledstates,1)*size(pooledstates,2)*size(pooledstates,3),1]));
return


