%This script is written to produce train and test file from a feature file
%%%%%input: %%%%%
%path: path of feature file
%filename: file name of feature file
%datasize:the size of the feature file
%the number of fold of crossvalidation
%%%%%%output:%%%%%
%train file and test file
function [trainfile, testfile]=producetrainandtestfile(path,filename,datasize,fold,randindex)
file=[path,filename];
fid=fopen(file,'r');
trainfile=[filename(1:end-4),'_train.dat'];
testfile=[filename(1:end-4),'_test.dat'];
fid1=fopen([path,trainfile],'w');
fid2=fopen([path,testfile],'w');

num=1;
while ~feof(fid)
    lines{num}=fgetl(fid);
    num=num+1;
end

testsize=floor(datasize*1/fold);
trainsize=datasize-testsize;


for i=1:trainsize
    line=lines{randindex(i)};
    fprintf(fid1,'%s\n',char(line));
end
for i=1:testsize
    line=lines{randindex(i)};
    fprintf(fid2,'%s\n',char(line));
end
fclose(fid2);
fclose(fid1);
fclose(fid);
end