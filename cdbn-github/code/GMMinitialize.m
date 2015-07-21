%This script is written to initialize the crbm by GMM model.
%So we have to get parameters from GMM: {w, u, covariance matrix}
%here, we suppose the covariance matrix to be sigma^2*I
%params:
%patches: in the form of [width,width,channels,numbers]
%numofhidunits: the number of hidden units
%output:[c,W,b] which initialize for 
%here, I implement a different one from the ICCV paper, I set visible layer
%bias to 0, which means c=0.

%by Dong Nie
%12/23/2014
function [c,W,b]=GMMinitialize(patchset,numofhidunits)
%initialize u with k-means
K=numofhidunits;
width=size(patchset,1);
channels=size(patchset,3);
num=size(patchset,4);
patches=reshape(patchset,[width*width*channels,num]);
d=width*width*channels;
[idx,C] = kmeans(patches',K);%the K centroids are in C, here C is [K*d]

%model=gmm_spherical_shared_cov(patches',C);
[label, model, llh] = emgm(patches, C');
sigma=sqrt(model.Sigma(1,1));
save('gmmmodel.mat','model');
%c=model.mu(:,1);
%W=1/sigma*(model.mu(:,2:size(model.mu,2))-repmat(c,[1,numofhidunits]));
% for i=2:length(model.weight)
%     b(i-1)=log(model.weight(i)/model.weight(1))-1/2*(norm(W(:,i-1),2))^2-1/sigma*W(:,i-1)'*c;
% end
c=0;
W=1/sigma*model.mu;
for i=1:length(model.weight)
    b(i)=log(model.weight(i))-1/2*(norm(W(:,i),2))^2;
end
W=reshape(W,[width,width,channels,numofhidunits]);
%c=reshape(c,[width,width,channels]);
return