%S是指隐层的state,根据隐层state和网络权重来重构visdata(大小还是还原成原始imdata大小了)
%其返回结果是sigma(conv2(feature_map,Weight))，应该是用来求P(v|h)的前奏,对negdata进行softmax函数化:real unit instead of binary unit)
%This script is written to reconstruct visible data using hidden layer
%data. here is a Gaussian RBM: P(v|h) follows N(sum(sigma*Wijhj+vbias),delta^2),
%so we just have to compute 1/delta*sum(Wijhj+hbias) where delta is noise
%of Gaussian. Here I rewrite the code to make it fit for multiple channels,
%and I also write it for computing more examples at a time.
%params
%hiddata:hidden layer states or probabilities, probabilites can speed up
%convergence process, so sometimes, we use probabilities
%hiddata:[hidpatchsize,hidpatchsize,numfilters,batchsize]
%W:[filtersize^2,numchannels,numfilters]
%output
%visdata:[patchsize,patchsize,numchannels,batchsize]
%Date:11/22/2014
%by: Dong Nie
function visdata = crbm_reconstruct_gaussian(hiddata, W, pars)
batchsize=size(hiddata,4);
filtersize = sqrt(size(W,1));
patch_M = size(hiddata,1);
patch_N = size(hiddata,2);
numchannels = size(W,2);
numfilters = size(W,3);

hiddata2 = hiddata;
visdata2 = zeros(patch_M+filtersize-1, patch_N+filtersize-1, numchannels,batchsize);%大小是:row(imdata)-filtersize+1,col(imdata)-filtersize+1

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
% visdata2 = zeros(patch_M+filtersize-1, patch_N+filtersize-1, numchannels,batchsize);%大小是:row(imdata)-filtersize+1,col(imdata)-filtersize+1
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

return
