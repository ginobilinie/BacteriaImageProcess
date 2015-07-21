function [a, grad] = log_act(X)

a = log(X);

if nargout > 1
    grad = X.^(-1);
end

end
