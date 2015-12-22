%This script is used to combine two feature files
%path='..\results\feature4SVMlayer2inputpooled_cdbnmodel1000\feature4SVM5000instance\bernoullifeatures\';
%path='..\results\maskImage\colwise_smallImage\';
path='..\results\maskImage\feature4 1-1-2 d databycdbn1000\feature41-1-2 d data bycdbn1000\bernoullifeatures\';
file1='layer1feature4svm.dat';
file2='layer3feature4svm.dat';
index=5184;
f=combinefeatures(path,file1,file2,index);