function [rf_index, pool_index, W, num_windows, h_dim, tied_units] = initialize_indices (param)

[rf_index, h_dim, num_windows] = initialize_rf_indices (param);

pool_index = initialize_pooling_indices (param, h_dim);

tied_units = initialize_tied_units (param, h_dim);

W = initialize_W (param, h_dim, rf_index);

end
