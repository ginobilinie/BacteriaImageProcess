function pars= paramsetup_layer2()
    pars.dataname='traindata';
    pars.inputType='gaussian';%2nd layer I use binary inputs
    pars.pbias=3e-3;%The sparsity target of layer1s
    pars.pbias_lambda=20;%The learning rate for sparsity, sparsity gain
    pars.spacing=2;
    pars.lRateStart=1e-3;
    pars.learningRate=pars.lRateStart;
    pars.lRateStop=1e-3;
    pars.l2reg=1e-4;%L2_regularization ¼´, wpenalty
    pars.numepochs=200;%important
    pars.numchannels=1;
    pars.initialmomentum  = 0.5;
    pars.finalmomentum    = 0.9;
    pars.sigma_start=2e-1;
    pars.sigma_stop=1e-1;
    pars.std_gaussian = pars.sigma_start;%
end
