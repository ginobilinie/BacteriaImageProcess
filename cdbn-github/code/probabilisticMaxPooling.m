function pooledFeatures = probabilisticMaxPooling(convolvedFeatures, strides)
%here note, when we use probabilistic max pooling, the pool layer's unit
%off probability is given by 1-sum(block in hidden layer), so the unit in
%pool layer's on's probabilisty is sum(block in hidden layer)
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
                pooledFeatures(floor(r/stride_row)+1,floor(c/stride_col)+1,i,k)=sum(sum(convolvedFeatures(r:r+stride_row-1,c:c+stride_col-1,i,k),1),2);
            end
        end
    end
end
end