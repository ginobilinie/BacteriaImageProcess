function [a, grad] = exp_act(X)

a = exp(X);

if nargout > 1
    grad = a;
end

end
