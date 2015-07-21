function cdbn=cdbnmodel(x1,pars,cdbn)
x=x1;
cdbn=cdbntrain(x,pars,cdbn);
return


function cdbn= cdbntrain(x, pars,cdbn)
    %n = numel(cdbn.crbm);%numofelement

    
    pars.filtersize=cdbn.crbm{1}.filtersize;
    pars.num_filters=cdbn.crbm{1}.num_filters;
    pars.numchannels=cdbn.crbm{1}.numchannels;
    pars.currentLayer=cdbn.crbm{1}.layer;
%     for i=1:10
%         for j=1:10
%             pars.pbias=0+i*0.01;
%             pars.pbias_lambda=1*j;
%             [crbm] = crbmtrain(x,cdbn.crbm{1},pars);%从v层->h1层
%             W1=crbm.W;
%              figure(1),display_network(W1);
%              saveas(gcf,sprintf('../results/cdbnvisual/std0.2_pbias%g_pbiaslamada%d_layer1.png',pars.pbias,pars.pbias_lambda));
%     %load('../results/cdbnmodel/crbmmodelby100_plamada0.03.mat','crbm');%to test
%             save(sprintf('../results/cdbnmodel/crbmmodel_pbias%g_pbias_lambda%g_by%d_layer1.mat',pars.pbias,pars.pbias_lambda,size(x,4)),'crbm');
%         end
%     end
%     pars.pbias=0.05;
%     pars.pbias_lambda=5;
%      load('initialsCRBM_layer1.mat','W','b');%this follows my personal inference
%      load('initialsCRBM_foreground06-Feb-2015_36filters_layer1.mat','W','b');%
    useGMMinitial4layer1=0;
    useTrainedCRBM4layer1=1;
    if useGMMinitial4layer1==1
    [c,W,b]=testGMMinitial();
     W=reshape(W,[size(W,1)^2,size(W,3),size(W,4)]);
%     W(:,:,7)=[];
%     b(:,7)=[];
     cdbn.crbm{1}.W=W;
     cdbn.crbm{1}.hbias_vec=b';
    end
    %load('../results/cdbnmodel/crbmmodel_learningRate0.001_l2reg0.0001_pbias0.001_pbias_lambda10_sigmastart0.3_sigmastop0.1_bytraindata200_06-Jan-2015_layer1_epoch500.mat','crbm');
    %cdbn.crbm{1}=crbm;
    size(x,4)
    if useTrainedCRBM4layer1==0
    [crbm] = crbmtrain(x,cdbn.crbm{1},pars);%从v层->h1层
    else
    load('../results/cdbnmodel/crbmmodel_learningRate0.001_l2reg0.0001_pbias0.002_pbias_lambda10_sigmastart0.3_sigmastop0.1_bytraindata100000_20-Feb-2015_layer1_epoch200.mat','crbm');
    end
    cdbn.crbm{1}=crbm;
    pars.std_gaussian = pars.sigma_stop;
    %pars=paramsetup();
    [poshidprobsFv,pooledprobsFV,poshidstatesFV]=getFeaturesByCRBMmodel(x,crbm,pars);
    pooledstatesFV=maxPooling(poshidstatesFV);
    save(sprintf('%s_poolingfeatures_36channels_layer.mat',date),'pooledprobsFV','pooledstatesFV');
    clear poshidstatesFV;%if not use this, clear it to save memory
    clear poshidprobsFv;
    W1=crbm.W;
    save(sprintf('../results/cdbnmodel/crbmmodel_learningRate%g_l2reg%g_pbias%g_pbias_lambda%g_sigmastart%g_sigmastop%g_bytraindata%d_%s_layer1.mat',pars.learningRate,pars.l2reg,pars.pbias,pars.pbias_lambda,pars.sigma_start,pars.sigma_stop,size(x,4),date),'crbm');
    figure(1),display_network_layer1(W1);
    saveas(gcf,sprintf('../results/cdbnvisual/learningRate%gl2reg%gpbias%gpbiaslamada%gsigmastart%gsigmastop%g%slayer1.png',pars.learningRate,pars.l2reg,pars.pbias,pars.pbias_lambda,pars.sigma_start,pars.sigma_stop,date));  
	
    
    %为了节省运行时间，我在这里只是当一张图片运行到epoch结束才显示figure的倍数时才显示
    %load('crbm_new1h_isp2_V1_w10_b100_p0.5_pl0.01_plambda2_sp2_CD_eps0.0001_l2reg1_bs03_20141005T022229_0030.mat','poshidstates');
    %pooledFeatures=maxPooling(crbm.poshidstates);%
    %cdbn.crbm{1}.pooledFeatures=pooledFeatures;
  %note, it is to test the parameters
%   for tt=1:10
%       for mm=1:10
%        pars.pbias=0.05+tt*0.05;%sparsity target for higher layer
%        pars.pbias_lambda=0+mm*1;%sparsity learning rate for higher layer
      % pars.std_gaussian=1;
        for i = 2 : cdbn.size
        %x = crbmup(dbn.crbm{i - 1}, x);%  rbmup其实就是简单的一句sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');也就是上面那张图从v到h计算一次，公式是Wx+c
            pars=paramsetup(2);%重新设立次参数,防止在程序中的底层被修改
            pars.filtersize=cdbn.crbm{i}.filtersize;
            pars.num_filters=cdbn.crbm{i}.num_filters;
            pars.numchannels=cdbn.crbm{i}.numchannels;
            pars.currentLayer=cdbn.crbm{i}.layer;
%             pars.pbias=0.05;%2nd layer
%             pars.pbias_lambda=3;
           
            
           % load('initialBMM_layer2_16filters.mat','W','b');%this follows my inference
            %W=reshape(W,[size(W,1)^2,size(W,3),size(W,4)]);
%             W(:,:,12)=[];
%             b(:,12)=[];
           % cdbn.crbm{2}.W=W;
            %cdbn.crbm{2}.hbias_vec=b';
            [crbm] = crbmtrain(pooledstatesFV,cdbn.crbm{i}, pars);%here I use pooledFeatures instead of poshidstates
            %load('../results/cdbnmodel/crbmmodel_learningRate0.001_l2reg0.0001_pbias0.005_pbias_lambda5_sigmastart0.2_sigmastop0.1_bytraindata200_07-Jan-2015_layer2_epoch200.mat','crbm');
            cdbn.crbm{i}=crbm;
            pars.std_gaussian = pars.sigma_stop;
			[poshidprobsFv,pooledprobsFV,poshidstatesFV]=getFeaturesByCRBMmodel(pooledstatesFV,crbm,pars);%lower layer's output as higher layer's input
			clear poshidstatesFV;%if not use, to save memory
            clear poshidprobsFv;
            %pooledFeatures=maxPooling(cdbn.crbm{i}.poshidstates);%here I use poshidprobs instead of poshidstates
            %cdbn.crbm{i}.pooledFeatures=pooledFeatures;
%         end
            W2=crbm.W;
            save(sprintf('../results/cdbnmodel/crbmmodel_learningRate%g_l2reg%g_pbias%g_pbias_lambda%g_sigmastart%g_sigmastop%g_bytraindata%d_%s_layer2.mat',pars.learningRate,pars.l2reg,pars.pbias,pars.pbias_lambda,pars.sigma_start,pars.sigma_stop,size(x,4),date),'crbm');
            figure(2),display_network_layer2(W2,W1);
        saveas(gcf,sprintf('../results/cdbnvisual/learningRate%gl2reg%gpbias%gpbiaslamada%gsigmastart%gsigmastop%g%slayer2.png',pars.learningRate,pars.l2reg,pars.pbias,pars.pbias_lambda,pars.sigma_start,pars.sigma_stop,date));
        end
        save(sprintf('../results/cdbnmodel/crbmmodel_learningRate%g_l2reg%g_pbias%g_pbiaslamada%d_cdbnmodel.mat',pars.learningRate,pars.l2reg,pars.pbias,pars.pbias_lambda),'cdbn');
%       end
%   end
    %now it comes to pooling

return

function cdbn = cdbnsetup(cdbn, x, opts)
    n = size(x, 2);% M = size(X,DIM) returns the length of the dimension specified;这里就是获取维度数 mnist_unit8是6w*784的数据
    cdbn.sizes = [n, cdbn.sizes];%1*2的矩阵，训练数据的维度数和100个隐藏节点

    tt=numel(cdbn.sizes);%取元素个数
    for u = 1 : numel(cdbn.sizes) - 1%这个循环我个人十分不理解
        cdbn.crbm{u}.alpha    = opts.alpha;%这里用了{}，表明rbm是个cell，学习率
        cdbn.crbm{u}.momentum = opts.momentum;%梯度下降时更新时的平滑参数

        %对于下面的代码:W、b、c是参数，vW、vb、vc是更新时与momentum的变量
        cdbn.crbm{u}.W  = zeros(cdbn.sizes(u + 1), dbn.sizes(u));%做了个hidden units*trainingset 维度数的矩阵
        cdbn.crbm{u}.vW = zeros(cdbn.sizes(u + 1), dbn.sizes(u));

        cdbn.crbm{u}.b  = zeros(cdbn.sizes(u), 1);
        cdbn.crbm{u}.vb = zeros(cdbn.sizes(u), 1);

        cdbn.crbm{u}.c  = zeros(cdbn.sizes(u + 1), 1);
        cdbn.crbm{u}.vc = zeros(cdbn.sizes(u + 1), 1);
    end

return



% function x = crbmup(crbm, x)
%     x = sigm(repmat(crbm.c', size(x, 1), 1) + x * crbm.W');
% return
% 
% function x = crbmdown(crbm, x)
%     x = sigm(repmat(crbm.b', size(x, 1), 1) + x * crbm.W);
% return



