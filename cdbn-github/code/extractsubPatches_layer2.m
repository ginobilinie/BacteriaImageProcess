%This script is written to extract subpatches from original patches (for
%example, 50*50), and these subpatches is usually to for initialization
function subf=extractsubPatches_layer2()
load('30-Jan-2015_poolingfeatures_36channels_layer.mat','pooledprobsFV')
D=10;
d=6;
step=1;

fpatches=pooledprobsFV;

len=min(20000000,size(fpatches,4));
cnt=0;
for i=1:len
    for j=1:step:D-d+1
        for k=1:step:D-d+1
            cnt=cnt+1;
            subf(:,:,:,cnt)=fpatches(j:j+d-1,k:k+d-1,:,i);
        end
    end
end

save(sprintf('%ssubpatches%d%dfrompooledlayer136channels%d%d.mat',date,D,D,d,d),'subf');
end
