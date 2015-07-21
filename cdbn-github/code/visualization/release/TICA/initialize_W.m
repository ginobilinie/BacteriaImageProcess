function [W] = initialize_W(param, h_dim, rf_index)
% Initialize W for normal local TICA
initialize_random_seeds(param.random_seed);

%W(1:param.num_maps*h_dim^2,:) = randn(param.num_maps*h_dim^2,param.input_ch*param.image_size^2);
W(1:param.num_maps*h_dim^2,:) = randn(param.num_maps*h_dim^2,param.image_size^2);
W = repmat(W, 1, param.input_ch); 
W = W.*rf_index;
W = shrink(rf_index,W);

end
