%This script is low quality, it is too slow
%here a is 2D, and B is 3D
function y = conv2_mult(a, B, convopt)
y = [];
for i=1:size(B,3)
    y(:,:,i) = conv2(a, B(:,:,i), convopt);
end
return