%This script is written to test the crbm program which turns out NaN after
%145 iterations, I store the 140 iteration data, and try to reboot the
%program from the 140th iteration
function W=crbm_test(beginning)
load crbm_new1h_isp2_V1_w10_b100_p0.5_pl0.01_plambda2_sp2_CD_eps0.0001_l2reg1_bs03_20141005T022229_0030.mat;
dataname='ISP2';
ws=10;
num_bases=100;
pbias=0.5;
pbias_lb=0.01;
pbias_lambda=2;
spacing=2;
epsilon=1e-4;
l2reg=1;%L2_regularization
batch_size=3;

pars=pars;
%function train_tirbm_updown_LB_v1h(dataname, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size)

if mod(ws,2)~=0, error('ws must be even number'); end

addpath ../../sparsenet/code
addpath ../../tisc/code/
addpath ../../poggio/code_new

sigma_start = 0.2;
sigma_stop = 0.1;

CD_mode = 'exp';
bias_mode = 'simple';

% Etc parameters
K_CD = 1;

% Initialization
%W = [];
%vbias_vec = [];
%hbias_vec = [];
pars = [];

C_sigm = 1;

% learning
num_trials = 500;

numchannels = 1;

% Initialize variables
if ~exist('pars', 'var') || isempty(pars)
    pars=[];
end

if ~isfield(pars, 'ws'), pars.ws = ws; end
if ~isfield(pars, 'num_bases'), pars.num_bases = num_bases; end
if ~isfield(pars, 'spacing'), pars.spacing = spacing; end

if ~isfield(pars, 'pbias'), pars.pbias = pbias; end
if ~isfield(pars, 'pbias_lb'), pars.pbias_lb = pbias_lb; end
if ~isfield(pars, 'pbias_lambda'), pars.pbias_lambda = pbias_lambda; end
if ~isfield(pars, 'bias_mode'), pars.bias_mode = bias_mode; end

if ~isfield(pars, 'std_gaussian'), pars.std_gaussian = sigma_start; end
if ~isfield(pars, 'sigma_start'), pars.sigma_start = sigma_start; end
if ~isfield(pars, 'sigma_stop'), pars.sigma_stop = sigma_stop; end

if ~isfield(pars, 'K_CD'), pars.K_CD = K_CD; end
if ~isfield(pars, 'CD_mode'), pars.CD_mode = CD_mode; end
if ~isfield(pars, 'C_sigm'), pars.C_sigm = C_sigm; end

if ~isfield(pars, 'num_trials'), pars.num_trials = num_trials; end
if ~isfield(pars, 'epsilon'), pars.epsilon = epsilon; end

disp(pars)


%% Initialize weight matrix, vbias_vec, hbias_vec (unless given)
% if ~exist('W', 'var') || isempty(W)
%     W = 0.01*randn(pars.ws^2, numchannels, pars.num_bases);
% end
% 
% if ~exist('vbias_vec', 'var') || isempty(vbias_vec)
%     vbias_vec = zeros(numchannels,1);
% end
% 
% if ~exist('hbias_vec', 'var') || isempty(hbias_vec)
%     hbias_vec = -0.1*ones(pars.num_bases,1);
% end


batch_ws = 70; % changed from 100 (2008/07/24)
imbatch_size = floor(100/batch_size);

fname_prefix = sprintf('../results/tirbm/tirbm_updown_LB_new1h_%s_V1_w%d_b%02d_p%g_pl%g_plambda%g_sp%d_CD_eps%g_l2reg%g_bs%02d_%s', dataname, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size, datestr(now, 30));
fname_save = sprintf('%s', fname_prefix);
fname_mat  = sprintf('%s.mat', fname_save);
fname_out = fname_mat;
mkdir(fileparts(fname_save));
fname_out

initialmomentum  = 0.5;
finalmomentum    = 0.9;

error_history = [];
sparsity_history = [];

Winc=0;
vbiasinc=0;
hbiasinc=0;

images_all = sample_images_all(dataname);%从文件夹里取出所有图像，并作初步处理

for t=beginning+1:pars.num_trials
    % Take a random permutation of the samples
    tic;
    ferr_current_iter = [];
    sparsity_curr_iter = [];

    len=length(images_all)
    imidx_batch = randsample(length(images_all), imbatch_size, length(images_all)<imbatch_size);
    %y = randsample(n,k,replacement) or y = randsample(population,k,replacement) returns a sample 
    %taken with replacement if replacement is true, or without replacement if replacement is false. The default is false.
    for i = 1:length(imidx_batch)%这里表示一次试验(num_trials)有多少批,共有100/3=33批次,每个批次是一张图片，这里其实是从1-6(试验中只有6张bacteria图片)做了置换，及33个数都是1-6的置换
        imidx = imidx_batch(i);
        imdata = images_all{imidx};
        rows = size(imdata,1);
        cols = size(imdata,2);

        for batch=1:batch_size%上面的一个imbatch,是一张图片，将一张图片subsample出3张70*70的patch,这里每个batch是一个70*70的patch
            % Show progress in epoch
            fprintf(1,'epoch %d image %d batch %d\r',t, imidx, batch); 

            rowidx = ceil(rand*(rows-2*ws-batch_ws))+ws + [1:batch_ws];%ws是卷积核大小
            colidx = ceil(rand*(cols-2*ws-batch_ws))+ws + [1:batch_ws];
            imdata_batch = imdata(rowidx, colidx);%从imdata（一张处理后的图片)取子patch
            imdata_batch = imdata_batch - mean(imdata_batch(:));%减去均值
            imdata_batch = trim_image_for_spacing_fixconv(imdata_batch, ws, spacing);%这个函数我没看太明白，估计是像java中trim字符串一样，去掉收尾的空格
            
            if rand()>0.5,
                imdata_batch = fliplr(imdata_batch);%fliplr函数实现矩阵左右翻转
            end
            
            % update rbm
            %下面这一句是核心啊,这里要注意:W,b(hbias_vec),c(vbias_vec)是全局变量,每次来一个patch，都要做一次rbm
            %更新,所以每次来个新的patch,{W,b,c}参数都随之更新
            %这里W已经有NaN value了
            if(sum(sum(sum(isnan(W))))>=1)
                fprintf('W have NaN value： %g\n',sum(sum(sum(isnan(W)))));
                W
            end
            [ferr dW dh dv poshidprobs poshidstates negdata]= fobj_tirbm_CD_LB_sparse(imdata_batch, W, hbias_vec, vbias_vec, pars, CD_mode, bias_mode, spacing, l2reg);
            ferr_current_iter = [ferr_current_iter, ferr];
            sparsity_curr_iter = [sparsity_curr_iter, mean(poshidprobs(:))];

            if t<5,
                momentum = initialmomentum;%0.5
            else
                momentum = finalmomentum;%0.9
            end

             if(sum(sum(sum(isnan(Winc))))>=1)
                fprintf('W have NaN value： %g\n',sum(sum(sum(isnan(Winc)))));
                Winc
             end
             
              if(sum(sum(sum(isnan(dW))))>=1)
                fprintf('W have NaN value： %g\n',sum(sum(sum(isnan(dW)))));
                dW
             end
            % update parameters
            Winc = momentum*Winc + epsilon*dW;%dw=momentum*dw_old+epsilon*dw
            W = W + Winc;
            
             if(sum(sum(sum(isnan(Winc))))>=1)
                fprintf('W have NaN value： %g\n',sum(sum(sum(isnan(Winc)))));
                Winc
             end
            
            if(sum(sum(sum(isnan(W))))>=1)
                fprintf('W have NaN value： %g\n',sum(sum(sum(isnan(W)))));
                W
            end
            
            vbiasinc = momentum*vbiasinc + epsilon*dv;%db..
            vbias_vec = vbias_vec + vbiasinc;

            hbiasinc = momentum*hbiasinc + epsilon*dh;%dc..
            hbias_vec = hbias_vec + hbiasinc;
        end
        mean_err = mean(ferr_current_iter);
        mean_sparsity = mean(sparsity_curr_iter);

        if (pars.std_gaussian > pars.sigma_stop) % stop decaying after some point
            pars.std_gaussian = pars.std_gaussian*0.99;
        end

        % figure(1), display_network(W);
        % figure(2), subplot(1,2,1), imagesc(imdata(rowidx, colidx)), colormap gray
        % subplot(1,2,2), imagesc(negdata), colormap gray
    end
    toc;

    error_history(t) = mean(ferr_current_iter);
    sparsity_history(t) = mean(sparsity_curr_iter);

    figure(1), display_network(W);
    % if mod(t,10)==0,
    %    saveas(gcf, sprintf('%s_%04d.png', fname_save, t));
    % end

    fprintf('epoch %d error = %g \tsparsity_hid = %g\n', t, mean(ferr_current_iter), mean(sparsity_curr_iter));
    save(fname_mat, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
    disp(sprintf('results saved as %s\n', fname_mat));
  
    if mod(t, 2) ==0
        fname_timestamp_save = sprintf('%s_%04d.mat', fname_prefix, t);
        save(fname_timestamp_save, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
    end

end%从t=1:num_trials到这里到这为止是一次试验，100个patch运行完毕

%return
return%crbm_test

%%

function images = sample_images_all(dataname)

switch lower(dataname),
case 'olshausen',
    fpath = '../data';
    flist = dir(sprintf('%s/*.tif', fpath));
case 'bacteria',
    fpath = '../bacteria';
    flist = dir(sprintf('%s/*.jpg', fpath));
 case 'isp2',
    fpath = '../ISP2';
    flist = dir(sprintf('%s/*.jpg', fpath));  
 case 'oatmeal',
    fpath = '../Oatmeal';
    flist = dir(sprintf('%s/*.jpg', fpath));
end

images = [];
for imidx = 1:min(length(flist), 200)
    fprintf('[%d]', imidx);
    fname = sprintf('%s/%s', fpath, flist(imidx).name);
    im = imread(fname);

    if size(im,3)>1
        im2 = double(rgb2gray(im));
    else
        im2 = double(im);
    end
    ratio = min([512/size(im,1), 512/size(im,2), 1]);
    if ratio<1
        im2 = imresize(im2, [round(ratio*size(im,1)), round(ratio*size(im,2))], 'bicubic');
    end

    im2 = tirbm_whiten_olshausen2(im2);
    % im2 = tirbm_whiten_olshausen1_contrastnorm(im2, 32, true, 0.6);
    % im2 = preprocess_image(im2, 'whiten'); % whiten the image before resizing
    im2 = im2-mean(mean(im2));
    im2 = im2/sqrt(mean(mean(im2.^2)));
    imdata = im2;
    imdata = sqrt(0.1)*imdata; % just for some trick??
    images{length(images)+1} = imdata;

    if 0
        imagesc(imdata), axis off equal, colormap gray
    end
end
fprintf('\n');


return


function [poshidexp2] = tirbm_inference(imdata, W, hbias_vec, pars)%计算P(h=1|V)

if (sum(sum(sum(isnan(W))))>=1)
    fprintf('W in tirbm_inference function has NaN value:%g\n',sum(sum(sum(isnan(W)))));
    W
end
ws = sqrt(size(W,1));
numbases = size(W,3);
numchannel = size(W,2);

poshidprobs2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);%初始化numbases个feature map(numbases个filter kernel, 即numbases个weight matrix,核的大小是ws*ws)
poshidexp2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
for c=1:numchannel
    H = reshape(W(end:-1:1, c, :),[ws,ws,numbases]);%将权重矩阵reshape成numbases个filter kernel
    poshidexp2 = poshidexp2 + conv2_mult(imdata(:,:,c), H, 'valid');%将filter kernel分别与输入数据(即imdata)做卷积
end

for b=1:numbases
    poshidexp2(:,:,b) = 1/(pars.std_gaussian^2).*(poshidexp2(:,:,b) + hbias_vec(b));
    poshidprobs2(:,:,b) = 1./(1 + exp(-poshidexp2(:,:,b)));
end
if (sum(sum(sum(isnan(poshidexp2))))>=1)
    fprintf('poshidprobs2 have %g NaN value\n',sum(sum(sum(isnan(poshidexp2)))));
end
if (sum(sum(sum(isnan(poshidprobs2))))>=1)
    fprintf('poshidprobs2 have %g NaN value\n',sum(sum(sum(isnan(poshidprobs2)))));
end
return

function [poshidprobs2,poshidexp2] = tirbm_inference_nie(imdata, W, hbias_vec, pars)%计算P(h=1|V)

if (sum(sum(sum(isnan(W))))>=1)
    fprintf('W in tirbm_inference function has NaN value:%g\n',sum(sum(sum(isnan(W)))));
    W
end
ws = sqrt(size(W,1));
numbases = size(W,3);
numchannel = size(W,2);

poshidprobs2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);%初始化numbases个feature map(numbases个filter kernel, 即numbases个weight matrix,核的大小是ws*ws)
poshidexp2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
for c=1:numchannel
    H = reshape(W(end:-1:1, c, :),[ws,ws,numbases]);%将权重矩阵reshape成numbases个filter kernel
    poshidexp2 = poshidexp2 + conv2_mult(imdata(:,:,c), H, 'valid');%将filter kernel分别与输入数据(即imdata)做卷积
end

for b=1:numbases
    %poshidexp2(:,:,b) = 1/(pars.std_gaussian^2).*(poshidexp2(:,:,b) + hbias_vec(b));
    poshidexp2(:,:,b) =(poshidexp2(:,:,b) + hbias_vec(b));
    poshidprobs2(:,:,b) = 1./(1 + exp(-poshidexp2(:,:,b)));
end
if (sum(sum(sum(isnan(poshidexp2))))>=1)
    fprintf('poshidprobs2 have %g NaN value\n',sum(sum(sum(isnan(poshidexp2)))));
end
if (sum(sum(sum(isnan(poshidprobs2))))>=1)
    fprintf('poshidprobs2 have %g NaN value\n',sum(sum(sum(isnan(poshidprobs2)))));
end
return


%这个函数是将V和H做卷积，其含义是求dE/dW，即权重矩阵的梯度
%在Algorithm1中代表Grad0
%H中有5个nan vlue
function vishidprod2 = tirbm_vishidprod_fixconv(imdata, H, ws)

numchannels = size(imdata,3);
numbases = size(H,3);

% tic
% TODO: single channel version is not implemented yet.. Might need to
% modify mexglx file
selidx1 = size(H,1):-1:1;
selidx2 = size(H,2):-1:1;
vishidprod2 = zeros(ws,ws,numchannels,numbases);

if numchannels==1
    vishidprod2 = conv2_mult(imdata, H(selidx1, selidx2, :), 'valid');%将imdata(就是输入数据)和隐层数据做卷积
else
    for b=1:numbases
        vishidprod2(:,:,:,b) = conv2_mult(imdata, H(selidx1, selidx2, b), 'valid');
    end
end

if(sum(sum(sum(sum(isnan(vishidprod2))))))%这里是因为H中有5个nan value造成的
    fprintf('There are %g nan value in vishidprod2',sum(sum(sum(sum(isnan(vishidprod2))))));
    vishidprod2
end

vishidprod2 = reshape(vishidprod2, [ws^2, numchannels, numbases]);%这里reshape后，得到100*100的矩阵，
%这里得到的只是将原始输入数据和隐层做卷积的矩阵，是在求梯度(dE/dW=V*H,*指代卷积操作)，这里的结果就是

return



%S是指隐层的state,根据隐层state和网络权重来重构visdata(大小还是还原成原始imdata大小了)
%这里是用CD来求dw的一个步骤，因为dw可以表达为vh-v'h',其中,vh叫postive stage,v'h'叫negative stage.
%这里就是明显的negative stage,当然，先要用gibbs sampling求出v(k)和h(k)
%其返回结果是sigma(conv2(feature_map,Weight))，应该是用来求P(v|h)的前奏,对negdata进行sigmoid函数化，然后加上bias就是V
function negdata = tirbm_reconstruct(S, W, pars)

ws = sqrt(size(W,1));
patch_M = size(S,1);
patch_N = size(S,2);
numchannels = size(W,2);
numbases = size(W,3);

S2 = S;
negdata2 = zeros(patch_M+ws-1, patch_N+ws-1, numchannels);%这是还原回来啊，还原到原来的卷积核大小,做feature map时，大小是:row(imdata)-ws+1,col(imdata)-ws+1

for b = 1:numbases,
    H = reshape(W(:,:,b),[ws,ws,numchannels]);
    negdata2 = negdata2 + conv2_mult(S2(:,:,b), H, 'full');
end

negdata = 1*negdata2;

return


%更新rbm函数，新来一个patch(imdata),更新网络参数{W,b,c}
function [ferr dW_total dh_total dv_total poshidprobs poshidstates negdata] = ...
    fobj_tirbm_CD_LB_sparse(imdata, W, hbias_vec, vbias_vec, pars, CD_mode, bias_mode, spacing, l2reg)

ws = sqrt(size(W,1));

%这里是在用CD方法求解RBM,即求解derivative{W,b,c},由于求导后的式子，第二项比较难求，如果只用gibbs
%sampling，那么可能要循环多次，即burn-in 时间太长。那么我们用CD，只用gibbs
%sampling中取k=1都可以，即此时只循环了一次就可以得到v'和h'，然后derivative{W}=vh-v'h'

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do convolution/ get poshidprobs
%我晕，这里W已经有NaN value了
poshidexp = tirbm_inference(imdata, W, hbias_vec, pars);%这里应该是找hidden
%layer的结果,即计算P(h=1|v),#######，原始代码
%[poshidprob_nie,poshidexp]=tirbm_inference_nie(imdata, W, hbias_vec, pars);%#####我改的代码
% poshidstates2 = double(poshidprobs > rand(size(poshidprobs))); 
%其实，这里poshidexp已经出现了NaN value,最终的NaN
%value是由于tirbm_sample_multrand2函数中exp(poshidexp)产生Inf造成的,
%所以，这里要找poshidexp防止其出现大于710的值,不知道作者这里为什么非要用poshidexp
[poshidstates poshidprobs] = tirbm_sample_multrand2(poshidexp, spacing);%spacing是卷积核移动步长，这个函数的意思是获取隐层的二值状态值和隐层的概率值,Bernoulli(PH0k)这个函数
%#####原始代码
%这里poshidprobs（或者说poshidexp）就是Algorithm2中的PH0,poshidstates就是Bernoulli化后的二值元组

%[poshidstates poshidprobs] = tirbm_sample_multrand2(poshidprob_nie, spacing);%#####我改的代码
%poshidstates = double(poshidprob_nie > rand(size(poshidprob_nie))); %#####我改的代码
%poshidprobs=poshidprob_nie;%#####我改的代码
if(sum(sum(sum(isnan(poshidprobs))))>=1)%isnan返回一个0,1数组
     disp('poshidprobs have nan value:\n');
     poshidprobs
end
%这里posprods在迭代过程中出现了nan,其实是poshidprobs中有nan造成的
posprods = tirbm_vishidprod_fixconv(imdata, poshidprobs, ws);%这里在权重矩阵的梯度(dE/dW,其操作是将V和H进行卷积操作)，即Algorithm2里的Grad0,

poshidact = squeeze(sum(sum(poshidprobs,1),2));%这个还不明白是干嘛的

if(sum(sum(sum(isnan(posprods))))>=1)%isnan返回一个0,1数组
     disp('posprods have nan value:\n');
     posprods
end

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%这里说白了，就是从hidden->visible layer呗，所谓reconstruction过程,求v'h'
neghidstates = poshidstates;
for j=1:pars.K_CD  %% pars.K_CD-step contrastive divergence
    negdata = tirbm_reconstruct(neghidstates, W, pars);%以隐层的二值状态和W来重构V,这里negdata只是feature map和Weight的卷积之和，求P(v|h)还需要sigmoid函数和bias
    % neghidprobs = tirbm_inference(negdata, W, hbias_vec, pars);
    % neghidstates = neghidprobs > rand(size(neghidprobs));
    neghidexp = tirbm_inference(negdata, W, hbias_vec, pars);%这是在算P(h|v)(algorithm2中V1(m)那一行?),其中negdata就是Algorithm2里的求V1(m)的表达式中sigmoid函数括号里的sigma和
    %#######原始代码
    [neghidstates neghidprobs] = tirbm_sample_multrand2(neghidexp, spacing);%再次获取隐层的二值状态值和概率值
    %######原始代码
    %[neghidprob_nie,neghidexp] = tirbm_inference_nie(negdata, W, hbias_vec, pars);%######我改的代码
    %[neghidstates neghidprobs] = tirbm_sample_multrand2(neghidprob_nie, spacing);%#####我改的代码
    %neghidstates = double(neghidprob_nie > rand(size(neghidprob_nie))); %#####我改的代码
    %neghidprobs=neghidprob_nie;%#####我改的代码
end
%这里negprods也在迭代过程中出现了nan
negprods = tirbm_vishidprod_fixconv(negdata, neghidprobs, ws);%将V和H做卷积，实质是求梯度Grad1
neghidact = squeeze(sum(sum(neghidprobs,1),2));%squeeze函数是去掉维度数为1的那个维度

ferr = mean( (imdata(:)-negdata(:)).^2 );%看来negdata就是被当做重构后的visible data(输入数据)

if 0
    figure(1), display_images(imdata)
    figure(2), display_images(negdata)

    figure(3), display_images(W)
    figure(4), display_images(posprods)
    figure(5), display_images(negprods)

    figure(6), display_images(poshidstates)
    figure(7), display_images(neghidstates)
end


%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if strcmp(bias_mode, 'none')
    dhbias = 0;
    dvbias = 0;
    dW = 0;
elseif strcmp(bias_mode, 'simple')
    dhbias = squeeze(mean(mean(poshidprobs,1),2)) - pars.pbias;
    dvbias = 0;
    dW = 0;
elseif strcmp(bias_mode, 'hgrad')
    error('hgrad not yet implemented!');
elseif strcmp(bias_mode, 'Whgrad')
    error('Whgrad not yet implemented!');
else
    error('wrong adjust_bias mode!');
end

numcases1 = size(poshidprobs,1)*size(poshidprobs,2);
% dW_total = (posprods-negprods)/numcases - l2reg*W - weightcost_l1*sign(W) - pars.pbias_lambda*dW;
dW_total1 = (posprods-negprods)/numcases1;%出现nan是因为posprods和negprods造成的
dW_total2 = - l2reg*W;
dW_total3 = - pars.pbias_lambda*dW;
dW_total = dW_total1 + dW_total2 + dW_total3;
if (isnan(dW_total(1,1,1)))
    fprintf('|dW|=%g, numcases1=%g\n',dW_total,numcases1);
end

dh_total = (poshidact-neghidact)/numcases1 - pars.pbias_lambda*dhbias;

dv_total = 0; %dv_total';

fprintf('||W||=%g, ||dWprod|| = %g, ||dWl2|| = %g, ||dWsparse|| = %g\n', sqrt(sum(W(:).^2)), sqrt(sum(dW_total1(:).^2)), sqrt(sum(dW_total2(:).^2)), sqrt(sum(dW_total3(:).^2)));

return



function [H HP] = tirbm_sample_multrand2(poshidexp, spacing)%这里HP明显有NaN值
% poshidexp is 3d array
%poshidprobs = exp(poshidexp);%有可能是这里产生的Inf,因为很明显，当poshidexp中某个元素大于709时，exp函数便会出现Inf
%######上面一句是原始代码
poshidprobs = exp(poshidexp-mean(poshidexp(:)));%#######我改的代码，减去均值，为了防止出现Inf的情况，
%这样不影响其后面均值化的值,唯一担心的就是作者在下面的poshidprobs_mult(end,:) =
%1;这样最后一列的1就变了,变得有可能有重要影响了
%poshidprobs=poshidexp;%#####我改的代码,到multirand2函数里再求exp(xx)，为了防止出现Inf的情况
if (sum(sum(sum(isnan(poshidexp))))>=1)
    fprintf('poshidexp has NaN value:%g',sum(sum(sum(isnan(poshidexp)))));
    poshidexp
end
if (sum(sum(sum(isnan(poshidprobs))))>=1)
    fprintf('poshidprobs has NaN value:%g',sum(sum(sum(isnan(poshidprobs)))));
    poshidprobs
end

if(sum(sum(sum(isinf(poshidexp))))>=1)
    fprintf('poshidexp have %g Inf value\n',sum(sum(sum(isinf(poshidexp)))));
    find(poshidexp==Inf)
end

if(sum(sum(sum(isinf(poshidprobs))))>=1)
    fprintf('poshidprobs have %g Inf value\n',sum(sum(sum(isinf(poshidprobs)))));
    find(poshidprobs==Inf)
end

poshidprobs_mult = zeros(spacing^2+1, size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,3)/spacing^2);
%poshidprobs_mult(end,:) = 1;%#####原始代码
poshidprobs_mult(end,:) =exp(-mean(poshidexp(:)));%我改的代码，为了与原来保持normalize完全一致才这么改，上面poshidprobs我已经在exp上减去均值了
% TODO: replace this with more realistic activation, bases..
for c=1:spacing
    for r=1:spacing
        temp = poshidprobs(r:spacing:end, c:spacing:end, :);
        poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
    end
end

%这里是因为poshidprobs_mult出现了Inf造成的,normalization时，Inf/Inf=NaN,所以我们要找到为什么会出现Inf
% [S P] = multrand2(poshidprobs_mult');
if(sum(sum(sum(isinf(poshidprobs_mult))))>=1)
    fprintf('poshidprobs_mult have %g Inf value\n',sum(sum(sum(isinf(poshidprobs_mult)))));
    find(poshidprobs_mult==Inf)
end
[S1 P1] = multrand2(poshidprobs_mult');%出问题的语句就是这个语句，所以要进到这个函数里去看个究竟
S = S1';
P = P1';
clear S1 P1

% convert back to original sized matrix
H = zeros(size(poshidexp));
HP = zeros(size(poshidexp));
for c=1:spacing
    for r=1:spacing
        H(r:spacing:end, c:spacing:end, :) = reshape(S((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
        HP(r:spacing:end, c:spacing:end, :) = reshape(P((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
    end
end
if(sum(sum(sum(isnan(HP))))>=1)
    fprintf('HP has %g NAN value\n',sum(sum(sum(isnan(HP)))));
    HP
end

return


function im2 = trim_image_for_spacing_fixconv(im2, ws, spacing)
% % Trim image so that it matches the spacing.
if mod(size(im2,1)-ws+1, spacing)~=0
    n = mod(size(im2,1)-ws+1, spacing);
    im2(1:floor(n/2), : ,:) = [];
    im2(end-ceil(n/2)+1:end, : ,:) = [];
end
if mod(size(im2,2)-ws+1, spacing)~=0
    n = mod(size(im2,2)-ws+1, spacing);
    im2(:, 1:floor(n/2), :) = [];
    im2(:, end-ceil(n/2)+1:end, :) = [];
end
return


function y = conv2_mult(a, B, convopt)
y = [];
for i=1:size(B,3)
    y(:,:,i) = conv2(a, B(:,:,i), convopt);
end
return


function test
if 0
%%
matlab -nodisplay
cd /afs/cs/u/hllee/visionnew/trunk/belief_nets/code_nips08
addpath tirbm

figure(1), display_network(W)
% figure(2), hist(W(:), 30)

pars.std_gaussian

t

%%
end
return

