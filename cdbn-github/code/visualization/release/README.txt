visualizeInvariances(network, gradFun, options, [additional arguments]) 
takes in a feedforward network object and produces videos of "invariant" transformations around the optimal stimulus of hidden units listed in options.visUnits

getOptimalStimulus(network, gradFun, options, [additional arguments]) is a routine that returns the optimal stimulus of units listed in options.visUnits.

Input arguments are:
	network: Feedforward network object

	gradFun: Gradient function 
		 gradFun(X, network, n, [additional arguments]) should return the gradient of the activation of hidden unit n with respect to the input at input X.
		 Note the [additional arguments] here are exactly the additional passed to visualizeInvariance.
		 This function is provided by the user, see the provided TICA example for illustration.

	options: Options for visualize invariances, this structure contains the following fields:
			networkName: Name of the network
		      cutoffPercent: % of optimal activation to set the cut off threshold for invariance at 
			     imSize: dimensions of the input image (2x1) vector
			   visUnits: Index of hidden units that should be visualized 
			     numDir: Number of eigenvectors to visualize 
			   rgbColor: produces 3-channel RGB output
			   saveOptX: save optimal input as a .mat file and visualizations of the optimal input as jpg files in savePath/images/
			  playVideo: plays back invariance videos after each one is generated (NOT RECOMMENDED IF LARGE NUMBERS OF VIDEOS ARE GENERATED)
			    savePath: where the output files should be saved (note invariance videos will be saved in savePath/videos/)
		
	[additional arguments]: Any additional arguments the user wishes to supply to gradFun


An example of using visualizeInvariances to visualize invariances of tiled convolutional neural networks is supplied, run the example by using the command visualizeTCNN('test_network.mat') an example of a gradFun function can also be found in the accompanying file TCNNgradFun.m
