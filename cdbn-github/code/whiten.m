function [patchset,precomp]=whiten(patchset,precomp)
% [patchset,precomp]=whiten(patchset,precomp)
% Performs whitening of patchset
% Returns whitened patchset and precomp structure storing information
% needed to whiten other patchsets.
%  Input:
%    patchset: d x n  n samples of d-dimensional patches
%    precomp: a structure storing information for whitening
%             precomp.mu - mean patch
%             precomp.W  - linear transformation that whitens
%             precomp.U  - inverse linear transformation that undoes
%                          whitening
%
% To visualize a gray whitened patch do:
%  imagesc(reshape(precomp.U*patchset(:,i) + precomp.mu,[sqrt(d) sqrt(d)]))
% 
patchset = single(patchset);
if nargin<2
    precomp = [];
end
if isempty(precomp)
    mu = mean(patchset,2);
else 
    mu = precomp.mu;
    W = precomp.W;
    U = precomp.U;
end

centered = bsxfun(@minus,patchset,mu);    
    
if isempty(precomp)
    lst = find(isfinite(sum(centered)));
    S = centered(:,lst)*centered(:,lst)'/length(lst);
    [E,D] = eig(S);
    dd = diag(D);
    lst = find(imag(dd)==0 & dd>1e-3);
    D = D(lst,lst); E = E(:,lst);
    W = E*diag(1./sqrt(diag(D)))*E';
    U = E*diag(sqrt(diag(D)))*E';
    precomp.W = W;
    precomp.U = U;
    precomp.mu = mu;
end

patchset = W*centered;
