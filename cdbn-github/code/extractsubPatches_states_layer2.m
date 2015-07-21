%This script is written to extract subpatches from original patches (for
%example, 50*50), and these subpatches is usually to for initialization
function subf=extractsubPatches_states_layer2()
load('24-Jan-2015_poolingfeatures_64channels_layer.mat','pooledstatesFV')
D=16;
d=10;
step=1;

fpatches=pooledstatesFV;

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

save(sprintf('%ssubpatches1010frompooledlayer164channels1616.mat',date),'subf');
end
