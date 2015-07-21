function pooledFeatures = maxPooling(convolvedFeatures, strides)
%这里我假设输入feature直接是 [row,col,nFM,batchsize]这种模式,strides就是一次走的步数，默认无重复
%由于时间紧张，我写的非常简单，不考虑除不开strides的问题,我一般用strides=[2,2]
if nargin==1
    strides=[2,2];%by default
end
nFM=size(convolvedFeatures,3);
row=size(convolvedFeatures,1);
col=size(convolvedFeatures,2);
batchsize=size(convolvedFeatures,4);
stride_row=strides(1);
stride_col=strides(2);

if (mod(row,stride_row)~=0||mod(col,stride_col)~=0)
    disp('stride cannot divide row or col completely\n');
end
pooledFeatures=zeros(row/stride_row,col/stride_col,nFM);
for k=1:batchsize
    for i=1:nFM
        for r=1:stride_row:row
            for c=1:stride_col:col
                pooledFeatures(floor(r/stride_row)+1,floor(c/stride_col)+1,i,k)=max(max(convolvedFeatures(r:r+stride_row-1,c:c+stride_col-1,i,k)));
            end
        end
    end
end
end