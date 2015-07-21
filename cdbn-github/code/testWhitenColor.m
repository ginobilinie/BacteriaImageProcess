patches = load('../patcheswithlabel/400rgbdata.mat');
patches = double(patchset);

figure(1);
displayColorNetwork(patches(:, 1:400));

numPatches=400;
epsilon=0.001;

% Scale data to range [0, 1]
patches = patches / 255;

% Subtract mean patch (hence zeroing the mean of the patches)
meanPatch = mean(patches, 2);
patches = bsxfun(@minus, patches, meanPatch);

% Apply ZCA whitening
sigma = patches * patches' / numPatches;
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ (diag(s) + epsilon)) * u';
patches = ZCAWhite * patches;
figure(2);
displayColorNetwork(patches(:, 1:400));
