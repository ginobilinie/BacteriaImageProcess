%here, I change the code to make it fit for 4-D data
%params:
%im2:[Heightpatchsize,Widthpatchsize,channels(colors),batchsize]
%filtersize: size of filters
%spacing: step to move
%Date:11/22/2014
%by: Dong Nie
function im2 = trim_image_for_spacing_fixconv(im2, filtersize, spacing)
% % Trim image so that it matches the spacing.
if mod(size(im2,1)-filtersize+1, spacing)~=0
    n = mod(size(im2,1)-filtersize+1, spacing);
    im2(1:floor(n/2), : ,:,:) = [];
    im2(end-ceil(n/2)+1:end, : ,:,:) = [];
end
if mod(size(im2,2)-filtersize+1, spacing)~=0
    n = mod(size(im2,2)-filtersize+1, spacing);
    im2(:, 1:floor(n/2), :,:) = [];
    im2(:, end-ceil(n/2)+1:end, :,:) = [];
end
return