%this function is used to combine separated feature data. When I extract
%feature from images using cdbn4images, it sometimes terminates at the
%middle of the process due to file size limit of os. So I try to combine
%the separated feature data together.
function [ outputfile ] = combineSepFeatureData()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
path='..\results\foreground\';
inputfile1=[path,'crbm_new1h_traindata_V1_w10_b36_p0.5_pl0.01_plambda2_sp2_CD_eps0.0001_l2reg1_bs03_20141008T234751_17383.mat'];
inputfile2=[path,'crbm_new1h_traindata_V1_w10_b36_p0.5_pl0.01_plambda2_sp2_CD_eps0.0001_l2reg1_bs03_20141009T135524_17289.mat'];
load(inputfile1,'cdbns')
cdbns1=cdbns;
load(inputfile2,'cdbns');
cdbns2=cdbns;
len1=length(cdbns1);
len2=length(cdbns2);
if strfind(path,'background')
    if len1>=len2
        disp('view the order of the input file...');
        return;
    end
    for i=1:len1
        if isempty(cdbns2{i})
            cdbns2{i}=cdbns1{i};
        end
    end
    cdbns=cdbns2;
    outputfile=[inputfile2,'_combine.mat'];
else%foreground
    index1=17383-1;
    index2=17289;
    for i=index1:-1:index2
        if isempty(cdbns1{i})
            cdbns1{i}=cdbns2{i};
        end
    end
    cdbns=cdbns1;
    outputfile=[inputfile2,'_combine.mat'];
end
save(outputfile,'cdbns');
end

