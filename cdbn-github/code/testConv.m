numchannel=3;
load X;
load Y;
load Z;
res=zeros(43,43,16);
for c=1:numchannel
    %H = reshape(W(end:-1:1, c, :),[filtersize,filtersize,numfilters]);%将权重矩阵reshape成numbases个filter kernel
    %poshidexp2 = poshidexp2 + conv2_mult(imdata(:,:,c), H, 'valid');%将filter kernel分别与输入数据(即imdata)做卷积

    W=reshape(Y,[64,3,16]);
    %H=Y(:,:,c,:);
    H = reshape(W(end:-1:1, c, :),[8,8,16]);
    H=squeeze(H);
    imdata = X(:,:,c);
    imdata=squeeze(imdata);
    conv2_mult(imdata,H,'valid');
    res=res+conv2_mult(imdata,H,'valid');
    
end

if max(res-Z)<1e-10
    disp('ok')
else
    disp('not ok');
end