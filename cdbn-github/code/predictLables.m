%This script is written to predict labels on testset, given testdata
%features, and traindata features

function pLabels=predictLables()
addpath('liblinear/');
traindatapath='../results/feature4SVM/allImages/features4SVMcoverAllImage_4layers.mat';
testdatapath='../results/feature4SVM/maskOneImage/features4firstImage_4layers.mat';


load(testdatapath,'layer1featuremat','trainedlabel');
testBetas=layer1featuremat;
testLabelMat=trainedlabel;


load(traindatapath,'layer1featuremat','trainedlabel');
trainLabelMat=trainedlabel;
trainBetas=layer1featuremat;

model = trainll(trainLabelMat',sparse(trainBetas'),'-s 6 -c 1 -v 2','row');
model = trainll(trainLabelMat',sparse(trainBetas'),'-s 6 -c 1','row');
%model = trainll(testLabelMat',sparse(testBetas'),'-s 6 -c 1 -v 5','row');
pLabels = predictll(testLabelMat',sparse(testBetas'),model,'','row');
%save([testdatapath,'layer3_result_bycdbn5000.mat'],'pLabels');
end