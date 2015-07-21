function [legal] = isLegal(v)
legal = sum(any(imag(v(:))))==0 & sum(isnan(v(:)))==0 & sum(isinf(v(:)))==0;
% if ~(legal)
%     disp('nan');
%     find(isnan(v))
%     disp('inf');
%     find(isinf(v))
%     disp('imaginary');
%     find(imag(v))
% end
