function [a, grad] = linear_act(x)

a = x;
if nargout > 1
        grad = ones(size(x));
end
