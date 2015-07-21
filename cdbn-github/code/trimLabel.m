%This script is written to trim the predicted labels: when positive labels are
%surrounded by negative labels, then this positive label should be
%negative, vice verse.
function [labels] = trimLabel(labelVec,numrow,numcol)
mat=reshape(labelVec,numrow,numcol);
for i=2:numrow-1
    for j=2:numcol-1
        if mat(i,j-1)==mat(i,j+1)&&mat(i-1,j)==mat(i+1,j)&&mat(i,j-1)==mat(i-1,j)%if surrounding are all the same
            mat(i,j)=mat(i,j-1);
        end
    end
end
labels=reshape(mat,[numrow*numcol,1]);
labels=squeeze(labels);
end

