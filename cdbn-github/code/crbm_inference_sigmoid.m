%This script is written to give a inference by sigmoid function (when the units are binary units, we should use sigmoid function)
%to compute fast, I add a dimension by batchsize, which can compute more
%example at the same time
%params
%imdata:[patchsize,patchsize,channels(colors),batchsize]
%W: filter:[filtersize^2,channels,numoffilters]
%output
%poshidprobs2:hidden layer units' probabilities' to be on:
%[hidsize,hidsize,numfilters]
%date:11/22/2014
%by: Dong Nie
function [poshidprobs2] = crbm_inference_sigmoid(imdata, W, hbias_vec, pars)%计算P(h=1|V)的第一步: I(hij)=hbias+W*V
batchsize=size(imdata,4);
filtersize = sqrt(size(W,1));
numfilters = size(W,3);
numchannel = size(W,2);

poshidprobs2 = zeros(size(imdata,1)-filtersize+1, size(imdata,2)-filtersize+1, numfilters,batchsize);%初始化numbases个feature map(numbases个filter kernel, 即numbases个weight matrix,核的大小是ws*filtersize)
poshidexp2 = zeros(size(imdata,1)-filtersize+1, size(imdata,2)-filtersize+1, numfilters,batchsize);

%one way to fulfill convolution is to do it for each example from the batchsize: but I have tested, this method is slower.
% tic;
% for n=1:batchsize%compute batch by batch
%     for c=1:numchannel%compute channel by channel%to include 3 color is now ok...
%         H = reshape(W(end:-1:1, c, :),[filtersize,filtersize,numfilters]);
%         poshidexp2(:,:,:,n) = poshidexp2(:,:,:,n) + conv2_mult(imdata(:,:,c,n), H, 'valid');%将filter kernel分别与输入数据(即imdata)做卷积
%     end
% end
% toc;
% v1=poshidexp2;
% poshidexp2 = zeros(size(imdata,1)-filtersize+1, size(imdata,2)-filtersize+1, numfilters,batchsize);
% fprintf('another version begin.\n');
% tic;
%another way to fulfill convolution, I believe this is much faster, especially when the batchsize is larger than 1. it is
%because I use convn instead of conv2
%tic
for k=1:numfilters
    for c=1:numchannel
        H = reshape(W(end:-1:1, c, k),[filtersize,filtersize]);
        poshidexp2(:,:,k,:) = poshidexp2(:,:,k,:) + convn(imdata(:,:,c,:), H, 'valid');%将filter kernel分别与输入数据(即imdata)做卷积
    end
end
%toc;
%v2=poshidexp2;

%here, the batchsize examples have the same hbais_vec 
for k=1:numfilters
    poshidexp2(:,:,k,:) = 1/(pars.std_gaussian^2).*(poshidexp2(:,:,k,:) + hbias_vec(k));
    poshidprobs2(:,:,k,:) = 1./(1 + exp(-poshidexp2(:,:,k,:)));%sigmoid, it is for lower layer
end

return
