function pars= paramsetup(layer)
if nargin==0
    layer=1;
end
if layer==1
    pars.dataname='traindata';
    pars.inputType='gaussian';
    pars.pbias=2e-3;%The sparsity target of layer1s
    pars.pbias_lambda=10;%The learning rate for sparsity, sparsity gain
    pars.spacing=2;
    pars.lRateStart=1e-3;
    pars.learningRate=pars.lRateStart;%这个就是学习率啊，learningRate
    pars.lRateStop=1e-3;
    pars.l2reg=1e-4;%L2_regularization 即, wpenalty
    pars.numepochs=200;%important
    pars.numchannels=1;
    pars.initialmomentum  = 0.5;
    pars.finalmomentum    = 0.9;
    pars.sigma_start=0.3;
    pars.sigma_stop=0.1;
    pars.std_gaussian = pars.sigma_start;%in second layer, it becomes 1
 
else
    pars=paramsetup_layer2();
end
pars.useGPU=0;
pars.useInternalGPU=0;%use GPU in inference and reconstruct funtion
pars.CD_mode ='exp';
pars.bias_mode = 'simple';
pars.K_CD = 1;
pars.traindatasize=100000;%it is important
pars.batchsize=500;%用batchsize大小来更新一次{w,b,c},这里用的是minibatch
end
