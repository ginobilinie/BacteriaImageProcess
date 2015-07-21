%This script is written to form featurevectors to fit SVM input format
%label fid:fvalue fid:fvalue....
function [bfeaturepath,mfeaturepath]=processfeaturevectors2SVM(featurevectors,y)
numofpatches=length(featurevectors);
if numofpatches~=length(y)
    disp('length of x and y are not matched\n');
    return;
end
path='..\results\feature4SVM\';

bfeaturepath=[path,'bernoullifeatures\'];%binary state features
boutputfile1=[bfeaturepath,'layer1feature4svm.dat'];
boutputfile2=[bfeaturepath,'layer2feature4svm.dat'];
boutputfile3=[bfeaturepath,'layer3feature4svm.dat'];
boutputfile4=[bfeaturepath,'layer4feature4svm.dat'];
bfid1=fopen(boutputfile1,'w');
bfid2=fopen(boutputfile2,'w');
bfid3=fopen(boutputfile3,'w');
bfid4=fopen(boutputfile4,'w');

mfeaturepath=[path,'multifeatures\'];%real valure features
moutputfile1=[mfeaturepath,'layer1feature4svm.dat'];
moutputfile2=[mfeaturepath,'layer2feature4svm.dat'];
% mfid1=fopen(moutputfile1,'w');
% mfid2=fopen(moutputfile2,'w');

token=' ';
eps=1e-3;%as poshidprobs are the value of p(h=1|v), so when it is two small, there is no meaning to count it as a feature

for i=1:numofpatches
    label=num2str(y(i));
    feature=featurevectors{i};
    
%     %first real value features:
%     mf1=feature{1}.poshidprobs;
%     s=[label,token];
%     for j=1:length(mf1)
%         if mf1(j)-eps>0%if the feature is too small, we count it as zero
%             s=[s,[[num2str(j),':'],longnum2shortstr(mf1(j))]];
%             s=[s,token];
%         end
%     end
%     fprintf(mfid1,'%s\n',s);
%     
%     mf2=feature{2}.poshidprobs;
%     s=[label,token];
%     for j=1:length(mf2)
%         if mf2(j)-eps>0%%if the feature is too small, we count it as zero
%             s=[s,[[num2str(j),':'],longnum2shortstr(mf2(j))]];
%             s=[s,token];
%         end
%     end
%     fprintf(mfid2,'%s\n',s);
    
    %second: binary features
    bf1=feature{1}.poshidstates;
    s=[label,token];
    for j=1:length(bf1)
        if bf1(j)~=0
            s=[s,[[num2str(j),':'],num2str(bf1(j))]];
            s=[s,token];
        end
    end
    fprintf(bfid1,'%s\n',s);
    
    bf2=feature{1}.pooledfeatures;
    s=[label,token];
    for j=1:length(bf2)
        if bf2(j)~=0
            s=[s,[[num2str(j),':'],num2str(bf2(j))]];
            s=[s,token];
        end
    end
    fprintf(bfid2,'%s\n',s);
    
    bf3=feature{2}.poshidstates;
    s=[label,token];
    for j=1:length(bf3)
        if bf3(j)~=0
            s=[s,[[num2str(j),':'],num2str(bf3(j))]];
            s=[s,token];
        end
    end
    fprintf(bfid3,'%s\n',s);
    
    bf4=feature{2}.pooledfeatures;
    s=[label,token];
    for j=1:length(bf4)
        if bf4(j)~=0
            s=[s,[[num2str(j),':'],num2str(bf4(j))]];
            s=[s,token];
        end
    end
    fprintf(bfid4,'%s\n',s);
end

% fclose(mfid1);
% fclose(mfid2);

fclose(bfid1);
fclose(bfid2);
fclose(bfid3);
fclose(bfid4);
end