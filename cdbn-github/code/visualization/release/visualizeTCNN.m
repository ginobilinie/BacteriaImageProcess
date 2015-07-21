function [] = visualizeTCNN(network_filename)
	% Include files needed for TICA feedforward
	tica_addpath;
	% Load TICA network
	load(network_filename);

	num_layers = numel(stacked_network.layer_W);
	image_size = stacked_network.network_params{1}.image_size;

	effectiveSize = 1;
	effectiveStep = 1;
	effectiveWindow = 1;

	% Find effective receptive field for top level units (i.e. local region of the image which it "looks" at)
	for a = 1:num_layers
		param = stacked_network.network_params{a};
		if isfield(param,'tied_size')
			param.tile_size = param.tied_size;
		end
		effectiveSize = effectiveSize + (param.window_size-1)*param.step + (param.pooling_size*2*param.poolstep);
		effectiveStep = effectiveStep * param.poolstep;

		[rf_index, pool_index, W, num_windows, h_dim, tied_units] = initialize_indices(param);
		W = expand_rf(param, h_dim, tied_units, stacked_network.layer_W{a});
		W = full_size(W, rf_index);

		effectiveWindow = double(stacked_network.layer_pool{a})*double(W~=0)*effectiveWindow;
	end
	
	% effectiveWindow contains the logical indexing of the receptive fields of all top layer hidden units
	effectiveWindow = logical(effectiveWindow~=0);

	% Grab a unit that sees a window of the full effective size (avoid of fringe effects)
	offset = floor((image_size - effectiveSize)/effectiveStep)*(floor(effectiveSize/effectiveStep)+1) + (floor(effectiveSize/effectiveStep)+1);
	[dummy, dummy, dummy, dummy, dummy, tied_units] = initialize_indices (param);
	visualized_units = [];
	for a = 1:numel(tied_units)
		visualized_units = [visualized_units,tied_units{a}(1) + offset];
	end

	% Set invariance visualization options
	options.visUnits = visualized_units(1:1);
	options.saveOptX = true;
	options.verbose = 1;
	[dummy, options.networkName] = fileparts(network_filename);
	options.savePath = './visualization_test/'
	options.imSize = [effectiveSize, effectiveSize*stacked_network.network_params{1}.input_ch];
	options.rgbColor = false;
	options.numDir = 3;
	% Generate optimal stimulus and invariance videos
	visualizeInvariances(stacked_network,@TCNNgradFun, options, effectiveWindow);
end
