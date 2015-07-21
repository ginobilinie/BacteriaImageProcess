function [c,W,b]= testGMMinitial()
layer=1;
%layer1
if layer==1
% gmmdatapath='22-Jan-2015-subpatches1818ffrom2000graypatch5050.mat';
% load(gmmdatapath);
[subf,subb]=extractsubPatches();
 n=size(subf,3);
% tt=subf;
 tt(:,:,size(subf,3)+1:size(subf,3)+size(subb,3))=subb;
 tt=reshape(tt,[size(tt,1),size(tt,2),1,size(tt,3)]);
K=[36];
for i=1:length(K)
 [c,W,b]=GMMinitialize(tt,K(i));
 save(sprintf('initialsCRBM_foreground%s_%dfilters_layer1.mat',date,K(i)),'c','W','b');
end
else
%layer2
%load('02-Jan-2015subpatches1010frompooled1stlayer1616.mat','sub');
subf=extractsubPatches_layer2();
tt=subf;
K=[9,16,25];
for i=1:length(K)
 [c,W,b]=GMMinitialize(tt,K(i));
 save(sprintf('initialsCRBM_foreground%s_%dfilters_layer2.mat',date,K(i)),'c','W','b');
end
end

end
