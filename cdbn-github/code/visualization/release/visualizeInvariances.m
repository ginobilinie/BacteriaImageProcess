function [] = visualizeInvariances(network,gradFun,options,varargin)
	
	addpath minConf/
	addpath minFunc/

	global logFile;

	[networkName, cutoffPercent, imSize, visUnits, numDir, rgbColor, saveOptX, playVideo, savePath] = processOptions(options);

	mkdir(savePath);
	mkdir([savePath '/videos/']);
	logFile = [savePath, 'visualization.log'];
		
	logPrint('Finding optimal stimulus for visualized units\n');
	% Find optimal stimulus
	[optXs] = getOptimalStimulus(network,gradFun,options,varargin{:});
	if saveOptX
		save([savePath, 'optX.mat'],'optXs');
	end

	% Visualize invariances
	for b = 1 : numel(visUnits)
		unit = visUnits(b);
		optX = optXs(:,b);
		tangent_basis = null(optX');

		% Numerically approximates hessian of activation given input using finite difference method
		% Projects hessian onto the tangent plane of the unit sphere at the optimal stimulus found
		H = getNumericalHessian(optX,network,gradFun,unit,varargin{:});
		H = tangent_basis'*H*tangent_basis;

		[V, D] = eig(H);
		test = real(diag(D));
		
		% Picks eigenvectors corresponding to the numDir least negative and most negative eigenvalues of H
		[test,ordering] = sort(test,'descend');
		ordering = ordering([1:numDir,length(ordering)-numDir+1:length(ordering)]);
		V = real(V(:,ordering));

		% Finds activation at optimal value of X and defined cutoff value for "invariance"
		[optAct] = gradFun(optX,network,unit,varargin{:});
		cutoff = cutoffPercent * optAct;

		% For each eigenvector chosen, "walk" in the chosen direction along the unit sphere until activation falls below cutoff
		for a = 1:length(ordering)
			movie = [];
			startT = 0;
			stopT = 0; 
			while startT >= -pi/2
				currX = optX * cos(startT) + sin(startT)*tangent_basis*V(:,a);
				currAct = gradFun(currX,network,unit,varargin{:});
				if currAct < cutoff
						startT = startT + pi/18;
						break;
				end
				startT = startT - pi/18;
			end
			while stopT <= pi/2
				currX = optX * cos(stopT) + sin(stopT)*tangent_basis*V(:,a);
				currAct = gradFun(currX,network,unit,varargin{:});
				if currAct < cutoff
						stopT = stopT - pi/18;
						break;
				end
				stopT = stopT + pi/18;
			end

			counter = 1;
			for t = startT:(pi/36):stopT
				currX = optX * cos(t) + sin(t)*tangent_basis*V(:,a);
				movie(:,:,counter) = reshape(currX,[imSize(1),imSize(2)]);
				counter = counter + 1;
			end

			% Normalize video for saving and display
			minval = min(min(min(min(movie))));
			movie = movie - minval;
			maxval = max(max(max(max(movie))));
			movie = movie / maxval;
			movie = uint8(round(255*movie));

			% Save invariance videos
			if (a<=numDir)
				moviefile = sprintf('%s/videos/%s_unit_%d_eig_low_%d.avi',savePath,networkName,unit,a);
			else
				moviefile = sprintf('%s/videos/%s_unit_%d_eig_high_%d.avi',savePath,networkName,unit,a-numDir);
			end

			imwidth = size(movie,2);

			for i = 1:size(movie,3)
				if rgbColor
					for j = 1:3
						temp = movie(:,(j-1)*(imwidth/3)+1:j*(imwidth/3),i);
						temp = imresize(temp,20);
						temp2(:,:,j) = temp;
					end
					avimovie(i) = im2frame(temp2);
				else
					temp = imresize(movie(:,:,i), 20);
					avimovie(i) = im2frame(temp,colormap(gray(256)));
				end
			end
			movie2avi(avimovie,moviefile,'fps',5);
			clear avimovie temp2;

			if playVideo
				implay(movie);
				pause;
			end
		end
	end
end
