%This scripts is written to train CDBN model (network parameters), then we
%can use this model to extract features from the train data and test data

function cdbn=traincdbnmodel(x,pars,cdbn)
%for lables:
%background=-1;
%forground=1;
if nargin==1
    cdbn=cdbnsetup();
    pars=paramsetup();
end

numofimages=size(x,2);
pars.numofbatches=floor(numofimages/pars.batchsize);

%traindataindex=randperm(numofimages,pars.traindatasize);
%save(sprintf('../results/cdbnmodel/traindataindex_%d_%s.mat',pars.traindatasize,date),'traindataindex');
pars.traindatasize=min(pars.traindatasize,size(x,2));
traindataindex=[1:pars.traindatasize];
x=x(:,traindataindex);

graypatchwidth=sqrt(size(x,1));%the initial input x is vetor form, here i input rgb patch
rgbpatchwidth=sqrt(size(x,1)/3);%rgb
if graypatchwidth==floor(graypatchwidth)%it is an gray image
    x=reshape(x,[graypatchwidth,graypatchwidth,1,size(x,2)]);% the third parameter is channel number
    cdbn.crbm{1}.numchannels=1;%if it is rgb:3, if it is gray 1
    pars.numchannels=1;
    disp('input is gray images\n');
elseif rgbpatchwidth==floor(rgbpatchwidth)%it is a rgb   
    x=reshape(x,[rgbpatchwidth,rgbpatchwidth,3,size(x,2)]);% the third parameter is channel number
    cdbn.crbm{1}.numchannels=3;%if it is rgb:3, if it is gray 1
    pars.numchannels=3;
    disp('input is rgb images\n');
else
    disp('input error\n');
end
%reinitialize the first layer's filters
cdbn.crbm{1}.W = 0.01*randn(cdbn.crbm{1}.filtersize^2, cdbn.crbm{1}.numchannels, cdbn.crbm{1}.num_filters);


cdbn=cdbnmodel(x,pars,cdbn);%a patch is a minipatch,as the train data is 500, a little small
fname_prefix=sprintf('../results/cdbnmodel/cdbn_%d_p%g__plambda%g_sp%d_CD_eps%g_l2reg%g_bs%02d_%s.mat',pars.traindatasize, pars.pbias, pars.pbias_lambda, pars.spacing, pars.learningRate, pars.l2reg, pars.batchsize, datestr(now, 30)');
fname_model_save = sprintf('%s_model_%04d.mat', fname_prefix, pars.traindatasize);
save(fname_model_save, 'cdbn');%存下当前的cdbn，即当前处理的那些照片的的特征结果,现在只是训练cdbn阶段，根本不用保存特征数据,

end
