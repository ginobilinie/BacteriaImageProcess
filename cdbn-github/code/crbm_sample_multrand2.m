%This script is the 2nd step to calculate the inference probability: in
%function crbm_inference_softmax, we get signals from lower layer: 
%I(hj)=sum(W*v)+bias; vice verse; this is the poshidexp parameter. Now I
%deal with the NaN problem, and then I make the code deal with batchsize
%examples at a time, this is very important: computational speed!
%params:
%poshidexp:signals from the lower layer:[hidsize,hidsize,numfilters,batchsize]
%spacing: move step
%output
%hidstates:hidden layer units' states:[hidsize,hidsize,numfilters,batchsize]
%hidprobs: hidden layer units' on probabilities:[hidsize,hidsize,numfilters,batchsize]
%Date:11/22/2014
%by: Dong Nie
function [hidstates hidprobs] = crbm_sample_multrand2(poshidexp, spacing)
% poshidexp is 3d array, now I add it to consider more examples a time
%poshidprobs = exp(poshidexp);%有可能是这里产生的Inf,因为很明显，当poshidexp中某个元素大于709时，exp函数便会出现Inf
%######上面一句是原始代码
%poshidprobs=poshidexp;%#####我改的代码,到multirand2函数里再求exp(xx)，为了防止出现Inf的情况,here is to get hij, then used to get probablistic form of hid layer(hidprobs)

batchsize=size(poshidexp,4);
hidstates = zeros(size(poshidexp));%bernoulli units in hidden layer
hidprobs = zeros(size(poshidexp));%p(h=1|v) in hidden layers

%This is not a good choice, to compute example by exmaple, it is slow
%tic;
for i=1:batchsize
    poshidprobs=poshidexp(:,:,:,i);
    poshidprobs_mult = zeros(spacing^2+1, size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,3)/spacing^2);%这里在实现probabilistic max-pooling
    %poshidprobs_mult(end,:) = 1;%#####原始代码
    poshidprobs_mult(end,:) = 0;%exp(0)=1我改的代码，好在multrand函数里统一exp化，这里其实是代表hidden layer一个以spacing为边长的block对应在的pooling layerP(pool=0|hidstates)

    % TODO: replace this with more realistic activation, bases..
    %下面要做的就是按照lee那篇论文中给出的求P(h(ij)=1|V)的方法，不是sigmoid函数，用的应该类似softmax
    %unit的方法，exp(I(h(ij)))/(1+sigma(exp(I(h)))),where h is a block where h(ij) is
    %contained in.

    %下面这个是将一个block(这里用的是2*2的block)里的所有unit放到一列来处理，由于还要考虑pooling
    %layer的P(pool=0|hidstates)，所以又加了一个,即spacing^2+1个unit,前spacing个代表hidden units
    %on的概率，后一个代表pooling layer off的概率
    %here, notice the area of the block is !!!!very important, it directly
    %affect max pooling!!!, it need to check again later.

    for c=1:spacing
        for r=1:spacing
            temp = poshidprobs(r:spacing:end, c:spacing:end, :);
            poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
        end
    end

    %这里是因为poshidprobs_mult出现了Inf造成的,normalization时，Inf/Inf=NaN,所以我们要找到为什么会出现Inf
    %to make it 
    [S1 P1] = multrand2(poshidprobs_mult');
    S = S1';
    P = P1';
    clear S1 P1

    % convert back to original sized matrix

    for c=1:spacing
        for r=1:spacing
            hidstates(r:spacing:end, c:spacing:end, :,i) = reshape(S((c-1)*spacing+r,:), [size(hidstates,1)/spacing, size(hidstates,2)/spacing, size(hidstates,3)]);
            hidprobs(r:spacing:end, c:spacing:end, :,i) = reshape(P((c-1)*spacing+r,:), [size(hidstates,1)/spacing, size(hidstates,2)/spacing, size(hidstates,3)]);
        end
    end
end
%toc;
return