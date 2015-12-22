function patches=testWhitenRGB(patches)
patches = double(patches) / 255;
meanPatch = mean(patches, 2);
patches = bsxfun(@minus, patches, meanPatch);

sigma = patches * patches';
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s))) * u';
patches = ZCAWhite * patches;
end