%This script is written to reconstruct visible data using hidden layer
%data. here is a Bernoulli-RBM: P(v|h)=sigmoid(sum(Wijhj+v_bias)),
%Here I rewrite the code to make it fit for multiple channels,
%and I also write it for computing more examples at a time. The most
%importatn thing is to speed up the computation
%params
%hiddata:hidden layer states [hidpatchsize,hidpatchsize,numfilters,batchsize]
%W:[filtersize^2,numchannels,numfilters]
%output
%visdata:[patchsize,patchsize,numchannels,batchsize]
%Date:11/22/2014
%by: Dong Nie
function visdata = crbm_reconstruct_sigmoid(hiddata, W, pars)
batchsize=size(hiddata,4);
filtersize = sqrt(size(W,1));
patch_M = size(hiddata,1);
patch_N = size(hiddata,2);
numchannels = size(W,2);
numfilters = size(W,3);

hiddata2 = hiddata;
visdata2 = zeros(patch_M+filtersize-1, patch_N+filtersize-1, numchannels,batchsize);%

%tic;
%after test, this is faster, So I use this
for k = 1:numfilters,
     H = reshape(W(:,:,k),[filtersize,filtersize,numchannels]);
     for i=1:batchsize
        visdata2(:,:,:,i) = visdata2(:,:,:,i) + conv2_mult(hiddata2(:,:,k,i), H, 'full');
    end
end
%toc;
%v1=visdata2;

%I donot know why, in this situation, to use convn is slower
% visdata2 = zeros(patch_M+filtersize-1, patch_N+filtersize-1, numchannels,batchsize);%¥Û–° «:row(imdata)-filtersize+1,col(imdata)-filtersize+1
% tic;
%  for b = 1:numfilters,
%         H = reshape(W(:,:,b),[filtersize,filtersize,numchannels]);
%         %tt=convn(hiddata2(:,:,b,:), H, 'full');;
%         visdata2(:,:,:,:) = visdata2(:,:,:,:) + convn(hiddata2(:,:,b,:), H, 'full');
%  end
%  toc;
%  v2=visdata2;
%

visdata = pars.std_gaussian*visdata2;
visdata=sigmoid(visdata);%mean field, converges fast
visdata=(visdata);%here I use real units, it converges fast
return
