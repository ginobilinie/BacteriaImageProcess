function [activations,temp_grad] = stack_tica_forwardprop(X, stacked_network)
layer_output = X;

num_layers = stacked_network.num_layers;
temp_grad = cell(num_layers,2);
activations = cell(num_layers,2);    

for a = 1:num_layers
    layer_param = stacked_network.network_params{a};
    [rf_index, pool_index, W, num_windows, h_dim, tied_units] = initialize_indices(layer_param);
 
    % Expand W
    W_temp = expand_rf(layer_param, h_dim, tied_units, stacked_network.layer_W{a});
    W_temp = full_size(W_temp, rf_index);
    
    % Forward prop 
    if nargout > 1       
         [activations{a,1} activations{a,2} temp_grad{a,1}, temp_grad{a,2}] = two_layer_forwardprop(layer_output, W_temp, stacked_network.layer_pool{a}, layer_param.l1_act, layer_param.l2_act);    
    else
         [activations{a,1} activations{a,2}] = two_layer_forwardprop(layer_output, W_temp, stacked_network.layer_pool{a}, layer_param.l1_act, layer_param.l2_act);
    end
    layer_output = activations{a,2};
end

end
