function [layer1, layer2, grad1, grad2] = two_layer_forwardprop(X, W, pool_layer, layer1_act, layer2_act)

if nargout <= 2
    layer1 = layer1_act(W*X);    
    clear W X;    
    l2_input = full((pool_layer)*double(layer1));    
    clear pool_layer;    
    layer2 = layer2_act(l2_input);
end

if nargout > 2 %only compute gradients if we need to    
    [layer1, grad1] = layer1_act(W*X);    
    clear W X;    
    l2_input = full((pool_layer)*double(layer1)); 
    clear pool_layer;    
    if nargout > 3
        [layer2, grad2] = layer2_act(l2_input);
    else
        layer2 = layer2_act(l2_input);
    end
end

end
