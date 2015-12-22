%This script is written to give an example how to run the whole
%system: train CDBN model, to get features, to train SVM, to see the
%classification accuracy of the model
%input:
%datapath: to get features from the data, datapath is the path of the
%inputdata file
%modelpath: the path of cdbn model
%output:
%cdbn:cdbn model file
%svmmodel: svm model file

function [cdbn,svmmodel] = doWholeProcess(datapath,modelpath)
%to load data first
load(datapath);
%trainFeatureMat
%trainLabelMat
%testFeatureMat
%testLabelMat

%define parameters for cdbn model
pars=paramsetup();
cdbn=cdbnsetup();

%%%%%%%%%%%%%%%%%%%%%%%%%%%train model%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%to train or load the cdbn model{network parameters}
if nargin==1% it means we have to train model first
    load(datapath);
%    train_x = double(reshape(train_x',28*28,60000))/255;
%     test_x = double(reshape(test_x',28*28,10000))/255;
%     train_y = double(train_y');
%     test_y = double(test_y');
 %    trainFeatureMat=train_x;
    %trainFeatureMat=double(trainFeatureMat)/255;
    fpatch=trainFeatureMat(:,find(trainLabelMat==1));
    bpatch=trainFeatureMat(:,find(trainLabelMat==-1));
    trainFeatureMat=[fpatch(:,:),bpatch(:,:)];
    cdbn=traincdbnmodel(trainFeatureMat,pars,cdbn);
    svmmodel=1;
    return;%end the program
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%learn features%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load(modelpath,'crbm');%I have train a cdbn model%only use 1000,12000
%to learn features using given cdbn model
% numofpatches=size(trainFeatureMat,2);
% traindatasize=10000;
% testdatasize=10000;
% traindataindex=randperm(numofpatches,traindatasize);
% traindataindex=1:traindatasize;%我晕，提取一张图片的特征时，是不能打乱train data的顺序的
% traindata=trainFeatureMat(:,traindataindex);


isTrain=1;%if it is to get features for the trained data or for the maskOne Image pictures
 model='crbm';%otherwise 'cdbn'
if (isTrain)
    num=20000;
    trainFeatureMat1=trainFeatureMat;
    trainLabelMat1=trainLabelMat;
    angles=[15,30,45,60,75,90];
    for angle=angles
    trainFeatureMat=rotatePatchset(trainFeatureMat1,angle);
    numofpatches=size(trainFeatureMat,2);
    patchindex=randperm(numofpatches,num);
    trainFeatureMat=trainFeatureMat(:,patchindex);
    trainLabelMat=trainLabelMat1(patchindex);
    %fpatch=trainFeatureMat(:,find(trainLabelMat==1));
    %bpatch=trainFeatureMat(:,find(trainLabelMat==-1));
    %traindata=[fpatch(:,1:num),bpatch(:,1:num)];
    x=trainFeatureMat;
    %trainedlabel=trainLabelMat;
    %trainedlabel=[ones(1,num),-1*ones(1,num)];
    if model=='crbm'
        graypatchwidth=sqrt(size(x,1));%the initial input x is vetor form, here i input rgb patch
        rgbpatchwidth=sqrt(size(x,1)/3);%rgb
        if graypatchwidth==floor(graypatchwidth)%it is an gray image
            x=reshape(x,[graypatchwidth,graypatchwidth,1,size(x,2)]);% the third parameter is channel number
            disp('input is gray images\n');
        elseif rgbpatchwidth==floor(rgbpatchwidth)%it is a rgb   
            x=reshape(x,[rgbpatchwidth,rgbpatchwidth,3,size(x,2)]);% the third parameter is channel number
            disp('input is rgb images\n');
        else
            disp('input error\n');
        end
        pars=paramsetup();
        pars.std_gaussian=pars.sigma_stop;
        [layer1features,layer1pooledfeatures,layer1states]=getFeaturesByCRBMmodel(x,crbm,pars);
        layer1pooledstates=maxPooling(layer1states);
        svmfeaturepath='../results/feature4SVM/allImages/';
        save([svmfeaturepath,sprintf('features4SVM_layer1_rotate%d.mat',angle)],'layer1features','layer1pooledfeatures','layer1pooledstates','trainLabelMat');%when use combine feature
    else
        [layer1features,layer1pooledfeatures,layer1pooledstates,layer2features,layer2pooledfeatures,layer2pooledstates]=getFeaturesByCDBNmodel(traindata,cdbn,pars);%This is fast, it takes about 9 minutes to extract features for 5,000 patches. 
        svmfeaturepath='../results/feature4SVM/allImages/';
        save([svmfeaturepath,'features4SVM_4layers.mat'],'layer1features','layer1pooledfeatures','layer1pooledstates','layer2features','layer2pooledfeatures','layer2pooledstates','trainLabelMat','specday');%when use combine feature
    end
    end
else
    x=trainFeatureMat;
    trainedlabel=trainLabelMat;
    if model=='crbm'
        graypatchwidth=sqrt(size(x,1));%the initial input x is vetor form, here i input rgb patch
        rgbpatchwidth=sqrt(size(x,1)/3);%rgb
        if graypatchwidth==floor(graypatchwidth)%it is an gray image
            x=reshape(x,[graypatchwidth,graypatchwidth,1,size(x,2)]);% the third parameter is channel number
            disp('input is gray images\n');
        elseif rgbpatchwidth==floor(rgbpatchwidth)%it is a rgb   
            x=reshape(x,[rgbpatchwidth,rgbpatchwidth,3,size(x,2)]);% the third parameter is channel number
            disp('input is rgb images\n');
        else
            disp('input error\n');
        end
        pars=paramsetup();
        pars.std_gaussian=pars.sigma_stop;
        [layer1features,layer1pooledfeatures,layer1states]=getFeaturesByCRBMmodel(x,crbm,pars);
        layer1pooledstates=maxPooling(layer1states);
        svmfeaturepath='../results/feature4SVM/maskOneImage/';
        save([svmfeaturepath,'features4SVM_layer1.mat'],'layer1features','layer1pooledfeatures','layer1pooledstates','trainLabelMat');%when use combine feature
    else
        [layer1features,layer1pooledfeatures,layer1pooledstates,layer2features,layer2pooledfeatures,layer2pooledstates]=getFeaturesByCDBNmodel(x,cdbn,pars);%This is fast, it takes about 9 minutes to extract features for 5,000 patches.
        svmfeaturepath='../results/feature4SVM/maskOneImage/';
        save([svmfeaturepath,'features4SVMcoverOneImage_4layers'],'layer1features','layer1pooledfeatures','layer1pooledstates','layer2features','layer2pooledfeatures','layer2pooledstates','trainedlabel','specday');%when use combine features, we can directly combine two matrices
    end
    
end
svmmodel=1;
%save([featurefilepath,'features4SVMcover5Images_4layers'],'layer1featuremat');
% % plabels=predictLables(trainfeatures,labels);
% % model = trainll(trainLabelMat',sparse(trainBetas'),'-s 6 -c 1','row');
% model1='layer1feature4svm_train.dat.model';
% model3='layer3feature4svm_train.dat.model';
% model13='combinefeatures4svm_train.dat.model';
% plab = predictll(testLabelMat',sparse(testBetas'),model,'','row');

end


