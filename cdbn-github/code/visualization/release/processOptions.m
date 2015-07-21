function [networkName, cutoffPercent, imSize, visUnits, numDir, rgbColor, saveOptX, playVideo, savePath] = processOptions(options)

        if isfield(options,'networkName')
                networkName = options.networkName;
        else
		networkName = '';
	end

        if isfield(options,'cutoff')
                cutoffPercent = options.cutoff;
        else
                cutoffPercent = 0.7;
        end
	
	if isfield(options,'imSize')
        	imSize = options.imSize;
	else
		error('Stimulus size not specified');
	end
	
	if isfield(options,'visUnits')
	        visUnits = options.visUnits;
	else
		error('Visualization units not specified');
	end

	if isfield(options,'savePath')
		savePath = options.savePath;
		if savePath(length(savePath)) ~= '/'
			savePath = [savePath, '/'];
		end
	else
		savePath = './';
	end

	savePath = [savePath,networkName,'/'];
	
	if isfield(options,'numDir')
		numDir = options.numDir;
	else
        	numDir = 10;
	end
	
	if isfield(options,'rgbColor')
		rgbColor = options.rgbColor;
	else
	        rgbColor = false;
	end
		
	if isfield(options,'saveOptX')
		saveOptX = options.saveOptX;
	else
		saveOptX = false;
	end

	if isfield(options,'playVideo')
		playVideo = options.playVideo;
	else
		playVideo = false;
	end
end
