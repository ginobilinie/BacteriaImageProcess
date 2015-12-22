%to imroate each patch in the patchset with a degree of angle
%param
%trainFeature: d*n
%angle: roate degree
function rmat=rotatePatchset(trainFeatureMat,angle)
    [d,n]=size(trainFeatureMat);
    grayd=sqrt(d);
    rgbd=sqrt(d/3);
    eps=10e-7;
    flag=1;%gray
    if (grayd-floor(grayd))<eps
        width=grayd;
        mat=reshape(trainFeatureMat,[width,width,n]);
    elseif (rgbd-floor(rgbd))<eps
        width=rgbd;
        mat=reshape(trainFeatureMat,[width,width,3,n]);
        flag=3;%rgbd
    end
    for i=1:n
        if flag==1
            patch=mat(:,:,i);
        else
            patch=mat(:,:,:,i);
        end
        rpatch=imrotate(patch,angle,'crop');
        rmat(:,i)=reshape(rpatch,size(rpatch,1)*size(rpatch,2)*size(rpatch,3),1);
    end
end