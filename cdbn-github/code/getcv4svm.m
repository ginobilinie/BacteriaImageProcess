%This script is written to call function to produce train and test file from a
%feature4svm file
%path,filename,datasize,fold
path='..\results\feature4SVMlayer2inputpooled_cdbmodel12000\feature4SVM15429instance\bernoullifeatures\';
% filename1='layer1feature4svm.dat';
% filename3='layer3feature4svm.dat';
% filename2='layer2feature4svm.dat';
% filename4='layer4feature4svm.dat';
filename='combinefeatures4svm.dat';
datasize=15429;
fold=5;
randindex=randperm(datasize);
[trainfile,testfile]=producetrainandtestfile(path,filename,datasize,fold,randindex);
% [trainfile1,testfile1]=producetrainandtestfile(path,filename1,datasize,fold,randindex);
% [trainfile3,testfile3]=producetrainandtestfile(path,filename3,datasize,fold,randindex);
% [trainfile2,testfile2]=producetrainandtestfile(path,filename2,datasize,fold,randindex);
% [trainfile4,testfile4]=producetrainandtestfile(path,filename4,datasize,fold,randindex);

%trainfile=[path,trainfile];%libsvm input format
%testfile=[path,testfile];

addpath('./libsvm/')
addpath('./liblinear/')
addpath('../liblinear-1.94/liblinear-1.94/windows');

%A matlab function libsvmread reads files in LIBSVM format: 
% [trainlabelvector, traininstancematrix] = libsvmread(trainfile); 
% [testlabelvector, testinstancematrix] = libsvmread(testfile);
% model = trainll(trainlabelvector,sparse(traininstancematrix),'-s 6 -c 1','row');
% [predict_label, accuracy] = predictll(testlabelvector,sparse(testinstancematrix),model,'row');

% model = train(trainlabelvector,sparse(traininstancematrix),'-s 6 -c 1','row');
% [predict_label, accuracy] = predict(testlabelvector,sparse(testinstancematrix),model,'row');