function [optXs] = getOptimalStimulus(network,gradFun,options,varargin)
% Takes in a network, gradient function and size of the input
% gradFun(X,network,n,varargin) should return the gradient of the activation of hidden node n wrt to input X

	global logFile;
	
	optXs = [];
	[dummy, dummy, imSize, visUnits, dummy, rgbColor, saveOptX, dummy, savePath] = processOptions(options);
	
        mkdir(savePath);
	mkdir([savePath '/images/']);
	if isempty(logFile)
	        logFile = [savePath, 'visualization.log'];
	end

	% minConf options
%	optOptions.Method = 'lbfgs';
	optOptions.method = 'lbfgs';
	optOptions.display = 'on';
	optOptions.verbose = 3;
	optOptions.TolX = 1e-10;
	optOptions.maxIter = 30;
	
	% Initialize X
	X = ones(prod(imSize),1); 
	logPrint('Finding optimal stimulus');
	clear stacked_network;

	% For each unit, maximise activation function with X constrained to lie within the unit sphere
	for a = visUnits
		tic
		logPrint('Optimizing unit: %d\n', a);
		anonGradFun = @(x) gradFunWrap(x,@(x)(gradFun(x,network,a,varargin{:})));
		[optX,optParams] = minConf_SPG(anonGradFun,X,@proj_norm,optOptions); 
		logPrint('Elapsed time: %f\n', toc);
		% Normalize and save optimal stimulus
		if saveOptX
			temp = reshape(optX,[imSize(1),imSize(2)]);
			temp = temp -min(min(temp));
			temp = temp ./ (max(max(temp)));
			imwrite(temp,sprintf('%s/images/unit%d.jpg',savePath,a),'jpg');
		end
		optXs = [optXs, optX(:)];
	end
end

function [grad,obj] = gradFunWrap(x, anonGradFun)
	[grad,obj] = anonGradFun(x);
	grad = -grad;
	obj = -obj;
end
	

