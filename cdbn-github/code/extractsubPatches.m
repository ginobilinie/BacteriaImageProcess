%This script is written to extract subpatches from original patches (for
%example, 50*50), and these subpatches is usually to for initialization
function [subf,subb]=extractsubPatches()
load('../../patcheswithlabel/data-11-Feb-2015Gray2828_sub4_step14_image1_158_pure_whiten.mat')
D=28;
d=8;
step=2;

size(trainFeatureMat,2)
%for foreground
fpatches=trainFeatureMat(:,trainLabelMat==1);
fpatches=reshape(fpatches,[D,D,size(fpatches,2)]);
%for background
bpatches=trainFeatureMat(:,trainLabelMat==-1);
bpatches=reshape(bpatches,[D,D,size(bpatches,2)]);

len=min(1000000,size(fpatches,3));
cnt=0;
for i=1:len
    for j=1:step:D-d+1
        for k=1:step:D-d+1
            cnt=cnt+1;
            subf(:,:,cnt)=fpatches(j:j+d-1,k:k+d-1,i);
        end
    end
end

len=min(1000000,size(bpatches,3));
cnt=0;
for i=1:len
    for j=1:step:D-d+1
        for k=1:step:D-d+1
            cnt=cnt+1;
            subb(:,:,cnt)=bpatches(j:j+d-1,k:k+d-1,i);
        end
    end
end

save(sprintf('%ssubpatches%d%dformgraypatch%d%d.mat',date,d,d,D,D),'subf','subb');
end
