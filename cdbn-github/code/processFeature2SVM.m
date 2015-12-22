%This script is written to deal with cdbns feature data, and extract
%features from the data which is fit for SVM light.
%data format:-1 1:0.43 3:0.12 9284:0.2 # abcdef
function [ outputfile ] = processFeature2SVM()
patchbegin=17289;
patchend=18070;
firstlayerFMrow=6;
firstlayerFMcol=6;
firstlayerFMnum=36;
secondlayerFMrow=2;
secondlayerFMcol=2;
secondlayerFMnum=100;
poolinglayerrow=1;
poolinglayercol=1;
poolinglayernum=100;
path='..\results\foreground\';
inputfile=[path,'crbm_new1h_traindata_V1_w10_b36_p0.5_pl0.01_plambda2_sp2_CD_eps0.0001_l2reg1_bs03_20141009T135524_17289.mat_combine.mat'];
load(inputfile,'cdbns');
len=length(cdbns);
label='1';
path=[path,'multifeatures\'];
outputfile1=[path,'layer1feature4foregrounddata.dat'];
outputfile2=[path,'layer2feature4foregrounddata.dat'];
outputfile3=[path,'layer3feature4foregrounddata.dat'];
fid1=fopen(outputfile1,'a+');
fid2=fopen(outputfile2,'a+');
fid3=fopen(outputfile3,'a+');



token=' ';
for i=patchbegin:patchend
    cdbn=cdbns{i};
    crbm1=cdbn.crbm{1};
    crbm2=cdbn.crbm{2};
    crbm3=cdbn.crbm{3};
    
    %f1=crbm1.poshidstate;
    f1=crbm1.poshidprobs;
    f1=reshape(f1,[firstlayerFMrow*firstlayerFMcol*firstlayerFMnum,1]);%6*6*36
    f1=squeeze(f1);
    s1=label;%label
    for j=1:length(f1)
        if f1(j)~=0
            s1=[s1,token];
            s1=[s1,[[num2str(j),':'],num2str(f1(j))]];
        end
    end
    fprintf(fid1,'%s\n',s1);
    
    %f2=crbm2.poshidstate;
    f2=crbm2.poshidprobs;
    f2=reshape(f2,[secondlayerFMrow*secondlayerFMcol*secondlayerFMnum,1]);
    f2=squeeze(f2);
    s2=label;%label
    for j=1:length(f2)
        if f2(j)~=0
            s2=[s2,token];
            s2=[s2,[[num2str(j),':'],num2str(f2(j))]];
        end
    end
    fprintf(fid2,'%s\n',s2);
    
    f3=crbm3.pooledFeatures;
    f3=reshape(f3,[poolinglayerrow*poolinglayercol*poolinglayernum,1]);
    f3=squeeze(f3);
    s3=label;%label
    for j=1:length(f3)
        if f3(j)~=0
            s3=[s3,token];
            s3=[s3,[[num2str(j),':'],num2str(f3(j))]];
        end
    end
    fprintf(fid3,'%s\n',s3);
end

fclose(fid1);
fclose(fid2);
fclose(fid3);
end

