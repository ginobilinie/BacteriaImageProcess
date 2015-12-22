% Example gradFun function, computes the gradient of the activation of unit ind wrt to the region of the local image it "looks" at
function[obj,grad] = TCNNgradFun(X, network, ind, effectiveWindow)

fullX = zeros(network.network_params{1}.image_size^2*network.network_params{1}.input_ch,size(X,2));
fullX(effectiveWindow(ind,:),:) = X;

layer_W = network.layer_W;
layer_pool = network.layer_pool;
num_layers = numel(layer_W);

% Forwardprop
[activations, grad] = stack_tica_forwardprop(fullX, network);

% Backprop
out_deriv = zeros(length(activations{num_layers,2}),1);
out_deriv(ind) = 1;
out_deriv = out_deriv(:,ones(size(grad{num_layers,2},2),1));
for a = num_layers : -1 : 1
    param = network.network_params{a};
    [rf_index, pool_index, W, num_windows, h_dim, tied_units] = initialize_indices(param);
    % backprop through 2nd activation function + pooling layer
    out_deriv = (full(layer_pool{a}))'*(out_deriv.*grad{a,2});
    % backprop through 1st activation function + W layer
    W_temp = expand_rf(param, h_dim, tied_units, layer_W{a});
    W_temp = full_size(W_temp, rf_index);
    W_temp = double(W_temp);
    out_deriv = (W_temp)'*(out_deriv.*grad{a,1});
end

grad = out_deriv(effectiveWindow(ind,:),:);
obj = activations{num_layers,2}(ind);

end

