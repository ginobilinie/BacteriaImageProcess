%This script is written to get features given a cdbn model(network
%parameters), and form the features for one object to a vector 
%cdbn:cdbn model 
%input:x is the patch, [row*col*channel,num]

function [layer1features,layer1pooledfeatures,layer1pooledstates,layer2features,layer2pooledfeatures,layer2pooledstates]=getFeaturesByCDBNmodel(x,cdbn,pars)
%load(fpath,'cdbn');
graypatchwidth=sqrt(size(x,1));%the initial input x is vetor form, here i input rgb patch
rgbpatchwidth=sqrt(size(x,1)/3);%rgb
if graypatchwidth==floor(graypatchwidth)%it is an gray image
    x=reshape(x,[graypatchwidth,graypatchwidth,1,size(x,2)]);% the third parameter is channel number
    disp('input is gray images\n');
elseif rgbpatchwidth==floor(rgbpatchwidth)%it is a rgb   
    x=reshape(x,[rgbpatchwidth,rgbpatchwidth,3,size(x,2)]);% the third parameter is channel number
    disp('input is rgb images\n');
else
    disp('input error\n');
end
pars=paramsetup();
pars.std_gaussian=pars.sigma_stop;
[poshidprobsFv,pooledprobsFV,poshidstatesFV]=getFeaturesByCRBMmodel(x,cdbn.crbm{1},pars);
layer1features=squeeze(reshape(poshidprobsFv,[size(poshidprobsFv,1)*size(poshidprobsFv,2)*size(poshidprobsFv,3),size(poshidprobsFv,4)])); 
layer1pooledfeatures=squeeze(reshape(pooledprobsFV,[size(pooledprobsFV,1)*size(pooledprobsFV,2)*size(pooledprobsFV,3),size(pooledprobsFV,4)])); 
pooledstatesFV=maxPooling(poshidstatesFV);
layer1pooledstates=pooledstatesFV;
clear poshidstatesFV;
clear poshidprobsFv;

pars=paramsetup(2);
pars.std_gaussian=pars.sigma_stop;
[poshidprobsFv,pooledprobsFV,poshidstatesFV]=getFeaturesByCRBMmodel(pooledprobsFV,cdbn.crbm{2},pars);
layer2features=squeeze(reshape(poshidprobsFv,[size(poshidprobsFv,1)*size(poshidprobsFv,2)*size(poshidprobsFv,3),size(poshidprobsFv,4)])); 
layer2pooledfeatures=squeeze(reshape(pooledprobsFV,[size(pooledprobsFV,1)*size(pooledprobsFV,2)*size(pooledprobsFV,3),size(pooledprobsFV,4)])); 
pooledstatesFV=maxPooling(poshidstatesFV);
layer2pooledstates=pooledstatesFV;
clear poshidstatesFV;
clear poshidprobsFv;

return

function [feature]=getfeature(x,cdbn,pars)
    layers=cdbn.size;%here we only care about convolution layer, pooling layer is stored in the lower layer
    imdata=x;
    if size(x,2)==1
        imdata=reshape(imdata,[sqrt(size(x,1)),sqrt(size(x,1))]);
    end
    for i=1:cdbn.size
        imdata = trim_image_for_spacing_fixconv(imdata, cdbn.crbm{i}.filtersize, pars.spacing);%trim
        W=cdbn.crbm{i}.W;
        hbias_vec=cdbn.crbm{i}.hbias_vec;
        vbias_vec=cdbn.crbm{i}.vbias_vec;
        % do convolution/ get poshidprobs
%         if i==1
            poshidexp = crbm_inference_softmax(imdata, W, hbias_vec, pars);%here get: I(hij)=hbias+W*V
            [poshidstates poshidprobs] = crbm_sample_multrand2(poshidexp, pars.spacing);%softmax(I(hij)) to get P(h=1|v) binary state 
%         else
%             poshidprobs = crbm_inference_sigmoid(imdata, W, hbias_vec, pars);%here get: I(hij)=hbias+W*V
%             poshidstates = double(poshidprobs > rand(size(poshidprobs))); 
%         end
        pooledprobs=probabilisticMaxPooling(poshidprobs);
        feature{i}.poshidstates=squeeze(reshape(poshidstates,[size(poshidstates,1)*size(poshidstates,2)*size(poshidstates,3),1])); 
        %feature{i}.poshidprobs=squeeze(reshape(poshidprobs,[size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,3),1]));
        
        feature{i}.pooledstates=squeeze(reshape(pooledstates,[size(pooledstates,1)*size(pooledstates,2)*size(pooledstates,3),1]));
        imdata=pooledstates;%pooledfeatures
    end
return


