function [H] = getNumericalHessian(X, network, gradFun, varargin)
	 epsilon = 1e-4;
	 
	 data = X(:,ones(numel(X),1)) + epsilon*eye(numel(X));
	 data = [X,data];

	 [obj,grad] = gradFun(data,network,varargin{:});

	 Hright = grad(:,2:end) - grad(:,ones(1,numel(X)));
	 Hright = Hright / epsilon;

	 data = X(:,ones(numel(X),1)) - epsilon*eye(numel(X));
	 data = [X,data];

	 [obj,grad] = gradFun(data,network,varargin{:});

	 Hleft = grad(:,2:end) - grad(:,ones(1,numel(X)));
	 Hleft = Hleft / (-epsilon);
 
	 H = (Hright + Hleft) / 2;	
	 H = (H + H') / 2;
end

