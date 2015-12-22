%更新crbm函数，新来一个patch(imdata),更新网络参数{W,b,c}
%In this script, We use CD(k=1) to calculate gradients, we also use pbias
%to reprenset the target sparsity in hidden layer, and we use l2reg to
%constraint the weight matrix to reduce overfitting
%here notice one thing: the input value is real data or binary!
%11/22/2014
%Dong Nie

function [ferr dW_total dh_total dv_total poshidprobs poshidstates recondata] = crbm_update_networks(imdata, W, hbias_vec, vbias_vec, pars)

filtersize = sqrt(size(W,1));

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% CRBM Up%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do convolution/ get poshidprobs
%%%%%%%real units layer%%%%%%%%%%%%%%%%%%%%%%
%if pars.currentLayer==1
    poshidexp = crbm_inference_softmax(imdata, W, hbias_vec, pars);%here I compute information from visible layer:I (hj)=sum(vi*Wij+h_bias)
    [poshidstates poshidprobs] = crbm_sample_multrand2(poshidexp, pars.spacing);%here I use softmax to compute P(h=1|v)=softmax(I);
    if strcmp(pars.CD_mode, 'mf'), poshidstates = poshidprobs; end %mean field update
% else%%%%%%%%%%binary units layer%%%%%%%%%%%%%%%%%%%
%     poshidprobs = crbm_inference_sigmoid(imdata, W, hbias_vec, pars);
%     poshidstates = double(poshidprobs > rand(size(poshidprobs))); 
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
posprods = crbm_vishidprod_fixconv(imdata, poshidprobs, filtersize);%to compute dw:sum(vi*p(hj|v0)):<vihj>data
poshidact = squeeze(sum(sum(sum(poshidprobs,1),2),4));%note,poshidprobs is [hidpatchsize hidpatchsize numfilters,batchsize], to compute dh

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% CRBM DOWN %%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sum(vihj):<vihj>model
neghidstates = poshidstates;% here it must use poshidstate, otherwise, it will not satisfisy the constraint
for j=1:pars.K_CD 
     if strcmp(pars.inputType,'gaussian')
        recondata = crbm_reconstruct_gaussian(neghidstates, W, pars);
        neghidexp = crbm_inference_softmax(recondata, W, hbias_vec, pars);%P(h|v)：sum(W*v)+h
        [neghidstates neghidprobs] = crbm_sample_multrand2(neghidexp, pars.spacing);%P(h|v):probability max-pooling
        if strcmp(pars.CD_mode, 'mf'), neghidstates = neghidprobs; end    
     elseif strcmp(pars.inputType,'binary')
        recondata = crbm_reconstruct_sigmoid(neghidstates, W, pars);%it is mean fild update, we donot make bernoulli distributed, because
        %we have to make it mean field update, which just get the expecation: E(v|h)=sigmoid(I);
        neghidprobs = crbm_inference_sigmoid(recondata, W, hbias_vec, pars);%just use probs when training
        neghidstates = neghidprobs > rand(size(neghidprobs));
        if strcmp(pars.CD_mode, 'mf'), neghidstates = neghidprobs; end
     else
         fprintf('inputType error\n');
    end
    
end
%vi hj from model 
negprods = crbm_vishidprod_fixconv(recondata, neghidprobs, filtersize);%E(vihj) follows model distribution
neghidact = squeeze(sum(sum(sum(neghidprobs,1),2),4));%neghidprobs:[hidpatchsize hidpatchsize numfilters batchsize]

ferr = mean( (imdata(:)-recondata(:)).^2 );%measure recon error

%now we have finished calculating two parts of derivatives:
%posprods->sum(vihj ) <vihj>data
%negprods->sum(vihj)  <vihj>model
%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if strcmp(pars.bias_mode, 'none')
    dhbias = 0;
    dvbias = 0;
    dW = 0;
elseif strcmp(pars.bias_mode, 'simple')
    dhbias = squeeze(mean(mean(mean(poshidprobs,1),2),4)) - pars.pbias;
    dvbias = 0;
    dW = 0;%here dW is for sparsity, while it hasn't been implemented
elseif strcmp(pars.bias_mode, 'hgrad')
    error('hgrad not yet implemented!');
elseif strcmp(pars.bias_mode, 'Whgrad')
    error('Whgrad not yet implemented!');
else
    error('wrong adjust_bias mode!');
end

numcases1 = size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,4);
% dW_total = (posprods-negprods)/numcases - l2reg*W - weightcost_l1*sign(W) - pars.pbias_lambda*dW;


dW_total1 = 1/pars.std_gaussian*(posprods-negprods)/numcases1;%average the gradient
dW_total2 = - pars.l2reg*W;
dW_total3 = - pars.pbias_lambda*dW;
dW_total = dW_total1 + dW_total2 + dW_total3;

dh_total = (poshidact-neghidact)/numcases1 - pars.pbias_lambda*dhbias;%update dh

dv_total = 0; %dv_total';

%fprintf('||W||=%g, ||dWprod|| = %g, ||dWl2|| = %g, ||dWsparse|| = %g\n', sqrt(sum(W(:).^2)), sqrt(sum(dW_total1(:).^2)), sqrt(sum(dW_total2(:).^2)), sqrt(sum(dW_total3(:).^2)));

return