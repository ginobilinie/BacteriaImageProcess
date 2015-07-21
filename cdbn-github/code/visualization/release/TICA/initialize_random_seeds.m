function initialize_random_seeds(seed)

randn('state', seed);
rand('state', seed);


