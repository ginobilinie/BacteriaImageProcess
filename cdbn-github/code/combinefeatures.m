%This script is wrtitten to combine lower layer features and higher layer
%features
%input: baisc path, feature filename, feature filename
%output:combined feature filename
function [f] = combinefeatures(path,featurefile1,featurefile2,index)
f1=[path,featurefile1];
f2=[path,featurefile2];
f=[path,'combinefeatures4svm.dat'];
fid1=fopen(f1,'r');
fid2=fopen(f2,'r');
fid=fopen(f,'w');
token=' ';
while ~feof(fid1)&&~feof(fid2)
    line1=fgetl(fid1);
    line2=fgetl(fid2);
    if line1(1)~=line2(1)
        disp('These two files are not consistent for the labels');
        return;
    end
    line=line1;
    S = regexp(line2, token, 'split');%label fid:fvalue fid:fvalue...
    for i=2:numel(S)%i=1 is label
        e=char(S{i});%fid:fvalue
        tt=regexp(e,':','split');
        if numel(tt)~=2%表示到结尾了
            break;
        end
        id=index+str2num(char(tt{1}));
        value=char(tt{2});
        ft=[[num2str(id),':'],value];
        ft=[ft,token];
        line=[line,ft];
    end
    fprintf(fid,'%s\n',line);
end
fclose(fid);
fclose(fid2);
fclose(fid1);
end

