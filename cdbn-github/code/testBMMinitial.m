% This file is from pmtk3.googlecode.com
%url:http://pmtk3.googlecode.com/svn/trunk/docs/demoOutput/bookDemos/%2811%29-Mixture_models_and_the_EM_algorithm/mixBerMnistEM.html
function testBMMinitial()
%setup data
% Ntest      = 1000;
% 
% if 1
%     Ntrain  = 1000;
%     Kvalues = [2, 10,16];
% else
%     Ntrain  = 1000;
%     Kvalues = 2:15;
% end
% % % [Xtrain, ytrain, Xtest, ytest] = setupMnist('binary', binary, 'ntrain',...
% % %     Ntrain,'ntest', Ntest,'keepSparse', keepSparse);
% % load('../../patcheswithlabel/mnist_uint8.mat');
% % Xtrain=double(train_x)/255;%train_x£º n*d
% % Xtest=double(test_x)/255;
% % Xtrain=Xtrain(1:Ntrain,:);
% % Xtest=Xtest(1:Ntest,:);
% % Xtrain=Xtrain>0.5;
% % Xtest=Xtest>0.5;
% % 
% % Xtrain = Xtrain + 1; % convert from 0:1 to 1:2
% % Xtest  = Xtest  + 1;
% 
% 
% load('17-Jan-2015subpatches1010frompooledstates1stlayer1616_35channels.mat','subf');
subf=extractsubPatches_states_layer2();
 width=size(subf,1);
 channel=size(subf,3);
 subf=reshape(subf,size(subf,1)*size(subf,2)*size(subf,3),size(subf,4));
 Xtrain=subf';%n*d
% 
 Xtrain=Xtrain+1;%0,1->1,2
% 
     Kvalues = [25,36,50,64,100];
% %Fit
 [n, d] = size(Xtrain);
 NK     = length(Kvalues);
% logp   = zeros(1, NK);
 %bicVal = zeros(1, NK);
 options = {'maxIter', 100, 'verbose', true};
 model = cell(1, NK);
 for i=1:NK%test different K values
     K = Kvalues(i);
     fprintf('Fitting K = %d \n', K)
     model{i}  = mixDiscreteFit(Xtrain, K, options{:});
% %     logp(i)   = sum(mixDiscreteLogprob(model{i}, Xtest));
% %     nParams   = K*d + K-1;
% %     bicVal(i) = -2*logp(i) + nParams*log(n);
 end

save(sprintf('bmm_layer2_foregroundpatches%s.mat',date),'model');
%load('bmm_layer2_mnist.mat','model');
%channels=35;
%width=10;
for k=1:NK
modelk=model{1,k};
c=0;
W= squeeze(modelk.cpd.T(:,2,:));
W=W';
for i=1:length(modelk.mixWeight)
    b(i)=log(modelk.mixWeight(i))-sum(log(1+exp(W(:,i))));
end
W=reshape(W,[width^2,channels,size(W,2)]);
save(sprintf('initialBMM_layer2_%dfilters%s.mat',Kvalues(k),date),'W','b','c');

end
%Plot
% for i=1:NK
%     K = Kvalues(i);
%     figure();
%     [ynum, xnum] = nsubplots(K);
%     if K==10
%         ynum = 2; xnum = 5;
%     end
%     if K==16
%         ynum=4; xnum=4;
%     end
%     TK = model{i}.cpd.T;
%     mixweightK = model{i}.mixWeight;
%     for j=1:K
%           group=reshape(TK(j, 2, :), [width*width,channel]);
%           display_network_layer1(group);
%         subplot(ynum, xnum, j);
%         imagesc(reshape(TK(j, 2, :), [width, width,channel]));
%         colormap('gray');
%         title(sprintf('%1.2f', mixweightK(j)), 'fontsize', 30);
%         axis off
%    end

 %   printPmtkFigure(sprintf('mixBernoulliMnist%d', K));
%end
% if numel(Kvalues) > 2
%     figure();
%     plot(Kvalues, bicVal, '-o', 'LineWidth', 2, 'MarkerSize', 8);
%     title(sprintf('Minimum achieved for K = %d', Kvalues(minidx(bicVal))));
%     printPmtkFigure('MnistBICvsKplot');
% end

return

function [Xtrain,ytrain,Xtest,ytest] = setupMnist(varargin)%binary, Ntrain, Ntest,full)
% Load mnist handwritten digit data
% Optional arguments [default in brackets]
% binary - if true, binarize around overall mean [false]
% ntrain - [60000]
% ntest - [10000]
% keepSparse - if true, do not cast to double [true]
% classes - specify which classes you want train/test data for [0:9]
%
% Xtrain will be ntrain*D, where D=784
% ytrain will be ntrain*1
% Xtest will be ntest*D, where D=784
% ytest will be ntest*1

% This file is from pmtk3.googlecode.com


[binary,Ntrain,Ntest,keepSparse,classes] = process_options(varargin,...
  'binary',false,'ntrain',60000,'ntest',10000,'keepSparse',true,'classes',0:9);
        
if nargout < 3, Ntest = 0; end

loadData('mnistAll');
% the datacase have already been shuffled 
% so we can safely take a prefix of the data
Xtrain = reshape(mnist.train_images(:,:,1:Ntrain),28*28,Ntrain)';
Xtest = reshape(mnist.test_images(:,:,1:Ntest),28*28,Ntest)';
ytrain = (mnist.train_labels);
ytest = (mnist.test_labels);
ytrain = ytrain(1:Ntrain);
ytest = ytest(1:Ntest);
clear mnist;
if(binary)
    mu = mean([Xtrain(:);Xtest(:)]);
    Xtrain = Xtrain >=mu;
    Xtest = Xtest >=mu;
end
ytrain = double(ytrain);
ytest  = double(ytest);

if(~keepSparse)
   Xtrain = double(Xtrain);
   Xtest  = double(Xtest);
end

if ~isequal(classes,0:9)
    Xtrain = Xtrain(ismember(ytrain,classes),:); 
    if numel(Ntest) > 0
       Xtest = Xtest(ismember(ytest,classes),:);
    end
end

return

function setSeed(seed)
% Set the random seed
% We don't use the new RandStream class for compatibility with Octave and
% older versions of Matlab. In the future it may be necessary to test the
% Matlab version and call the appropriate code. 

% This file is from pmtk3.googlecode.com

global RNDN_STATE  RND_STATE
if nargin == 0
    seed = 0; 
end
warning('off', 'MATLAB:RandStream:ReadingInactiveLegacyGeneratorState');
RNDN_STATE = randn('state');  %#ok<*RAND>
randn('state', seed);
RND_STATE = rand('state');
rand('twister', seed);
return

%% Process named arguments to a function
%
% This allows you to pass in arguments using name, value pairs
% eg func(x, y, 'u', 0, 'v', 1)
% Or you can pass in a  struct with named fields
% eg  S.u = 0; S.v = 1; func(x, y, S)
%
% Usage:  [var1, var2, ..., varn[, unused]] = ...
%           process_options(args, ...
%                           str1, def1, str2, def2, ..., strn, defn)
%
%
% Arguments:   
%            args            - a cell array of input arguments, such
%                              as that provided by VARARGIN.  Its contents
%                              should alternate between strings and
%                              values.
%            str1, ..., strn - Strings that are associated with a 
%                              particular variable
%            def1, ..., defn - Default values returned if no option
%                              is supplied
%
% Returns:
%            var1, ..., varn - values to be assigned to variables
%            unused          - an optional cell array of those 
%                              string-value pairs that were unused;
%                              if this is not supplied, then a
%                              warning will be issued for each
%                              option in args that lacked a match.
%
% Examples:
%
% Suppose we wish to define a Matlab function 'func' that has
% required parameters x and y, and optional arguments 'u' and 'v'.
% With the definition
%
%   function y = func(x, y, varargin)
%
%     [u, v] = process_options(varargin, 'u', 0, 'v', 1);
%
% calling func(0, 1, 'v', 2) will assign 0 to x, 1 to y, 0 to u, and 2
% to v.  The parameter names are insensitive to case; calling 
% func(0, 1, 'V', 2) has the same effect.  The function call
% 
%   func(0, 1, 'u', 5, 'z', 2);
%
% will result in u having the value 5 and v having value 1, but
% will issue a warning that the 'z' option has not been used.  On
% the other hand, if func is defined as
%
%   function y = func(x, y, varargin)
%
%     [u, v, unused_args] = process_options(varargin, 'u', 0, 'v', 1);
%
% then the call func(0, 1, 'u', 5, 'z', 2) will yield no warning,
% and unused_args will have the value {'z', 2}.  This behaviour is
% useful for functions with options that invoke other functions
% with options; all options can be passed to the outer function and
% its unprocessed arguments can be passed to the inner function.
%

% This file is from pmtk3.googlecode.com


%PMTKauthor Mark Paskin
%PMTKurl http://ai.stanford.edu/~paskin/software.html
%PMTKmodified Matt Dunham 

% Copyright (C) 2002 Mark A. Paskin

function [varargout] = process_options(args, varargin)

args = prepareArgs(args); % added to support structured arguments
% Check the number of input arguments
n = length(varargin);
if (mod(n, 2))
  error('Each option must be a string/value pair.');
end

% Check the number of supplied output arguments
if (nargout < (n / 2))
  error('Insufficient number of output arguments given');
elseif (nargout == (n / 2))
  warn = 1;
  nout = n / 2;
else
  warn = 0;
  nout = n / 2 + 1;
end

% Set outputs to be defaults
varargout = cell(1, nout);
for i=2:2:n
  varargout{i/2} = varargin{i};
end

% Now process all arguments
nunused = 0;
for i=1:2:length(args)
  found = 0;
  for j=1:2:n
    if strcmpi(args{i}, varargin{j}) || strcmpi(args{i}(2:end),varargin{j})
      varargout{(j + 1)/2} = args{i + 1};
      found = 1;
      break;
    end
  end
  if (~found)
    if (warn)
      warning(sprintf('Option ''%s'' not used.', args{i}));
      args{i}
    else
      nunused = nunused + 1;
      unused{2 * nunused - 1} = args{i};
      unused{2 * nunused} = args{i + 1};
    end
  end
end

% Assign the unused arguments
if (~warn)
  if (nunused)
    varargout{nout} = unused;
  else
    varargout{nout} = cell(0);
  end
end

return

function out = prepareArgs(args)
% Convert a struct into a name/value cell array for use by process_options
%
% Prepare varargin args for process_options by converting a struct in args{1}
% into a name/value pair cell array. If args{1} is not a struct, args
% is left unchanged.
% Example:
% opts.maxIter = 100;
% opts.verbose = true;
% foo(opts)
% 
% This is equivalent to calling 
% foo('maxiter', 100, 'verbose', true)

% This file is from pmtk3.googlecode.com


if isstruct(args)
    out = interweave(fieldnames(args), struct2cell(args));
elseif ~isempty(args) && isstruct(args{1})
    out = interweave(fieldnames(args{1}), struct2cell(args{1}));
else
    out = args;
end

return


function [model, loglikHist] = mixDiscreteFit(data, nmix, varargin)
%% Fit a mixture of product of multinoullis via MLE/MAP (using EM)
%
% By default we lightly regularize the parameters, so we are doing map
% estimation. To turn this off, set 'prior' and 'mixPrior to 'none'. See
% Inputs below.
%
%% Inputs
%
% data     - data(n, :) is the ith case, i.e. data is of size N*D
%  We current require that data(n, d) in {1...C} where
%  C is the same for all dimensions
% nmix     - the number of mixture components to use
% alpha     - value of Dirichlet prior on observations, default 1.1 (1=MLE)
% EMargs - cell arrya, see emAlgo


% This file is from pmtk3.googlecode.com

[initParams, mixPrior, alpha,  verbose, EMargs] = ...
    process_options(varargin, ...
    'initParams'        , [], ...
    'mixPrior', [], ...
    'alpha', 1.1, ...        
    'verbose', true);
     
[n, d]      = size(data);
model.type  = 'mixDiscrete';
model.nmix  = nmix;
model.d     = d;
model       = setMixPrior(model, mixPrior );
prior.alpha = alpha;
initFn = @(m, X, r)initDiscrete(m, X, r, initParams, prior); 
[model, loglikHist] = emAlgo(model, data, initFn, @estep, @mstep , ...
                            'verbose', verbose, EMargs{:});
return


function model = initDiscrete(model, X, restartNum, initParams, prior)
%% Initialize
nObsStates = max(nunique(X(:)));
if restartNum == 1 && ~isempty(initParams)
    T = initParams.T;
    model.mixWeight = initParams.mixWeight;
else
    % randomly partition data, fit each partition separately, add noise.
    nmix    = model.nmix;
    d       = size(X, 2);
    T       = zeros(nmix, nObsStates, d);
    Xsplit  = randsplit(X, nmix);
    for k=1:nmix
        m = discreteFit(Xsplit{k}, 1, nObsStates);
        T(k, :, :) = m.T;
    end
    T               = normalize(T + 0.2*rand(size(T)), 2); % add noise
    model.mixWeight = normalize(10*rand(1, nmix) + ones(1, nmix));
end
model.cpd = condDiscreteProdCpdCreate(T, 'prior', prior); 
return


function [ess, loglik] = estep(model, data)
%% Compute the expected sufficient statistics
[weights, ll] = mixDiscreteInferLatent(model, data); 
cpd           = model.cpd;
ess           = cpd.essFn(cpd, data, weights); 
ess.weights   = weights; % useful for plottings
loglik        = sum(ll) + cpd.logPriorFn(cpd) + model.mixPriorFn(model); 
return

function model = mstep(model, ess)
%% Maximize
cpd             = model.cpd;
model.cpd       = cpd.fitFnEss(cpd, ess); 
model.mixWeight = normalize(ess.wsum + model.mixPrior - 1); 
return


function model = setMixPrior(model, mixPrior)
%% Set the mixture prior
nmix = model.nmix; 
if isempty(mixPrior)
    model.mixPrior = 2*ones(1, nmix);
end
if ischar('none') && strcmpi(mixPrior, 'none'); 
    model.mixPrior = ones(1, nmix);
    model.mixPriorFn = @(m)0;
else
    model.mixPriorFn  = @(m)log(m.mixWeight(:))'*(m.mixPrior(:)-1);
end
if isscalar(model.mixPrior)
    model.mixPrior = repmat(model.mixPrior, 1, nmix); 
end
return

function [X, y] = mixDiscreteSample(T, mixWeight, nsamples)
% Sample nsamples from a mixture of discrete distributions. 
% X(i, :) is the ith sample generated from mixture y(i). 
%
%% INPUTS:
%
% T is a matrix of size nmix-nObsStates-d
% mixWeight is a stochastic vector of size 1-by-nmix
%% OUTPUTS:
% X is of size nsamples-by-d and X(i,j) is in the range 1:nObsStates
% y is of size nsamples-by-1 and y(i) is in the range 1:nmix
%
%%

% This file is from pmtk3.googlecode.com

[nmix, nObsStates, d] = size(T); 
y = sampleDiscrete(mixWeight, nsamples, 1); 
X = zeros(nsamples, d);

for i=1:nsamples
    for j=1:d
        X(i, j) = sampleDiscrete(T(y(i), :, j), 1);
    end
end
return

function [pX] = mixDiscretePredictMissing(model, X)
% Compute pX(i,j,v) = p(Xj=v|case i), where X(i,:) may be partially
% observed (with NaNs)
% This is like mixModelReconstruct expect we compute the posterior
% predictive distribution, intsead of using MAP estimation

% This file is from pmtk3.googlecode.com


[pZ] = mixDiscreteInferLatent(model, X); % pZ is Ncases*Nnodes
[Ncases Nnodes] = size(pZ); %#ok
[Nstates, NobsStates, ndims] = size(model.cpd.T); %#ok
%pX = zeros(Ncases, NobsStates, Nnodes);


% First make delta functions for observed entries
nStates  = NobsStates*ones(1,ndims);
[~, pX] = dummyEncoding(X, nStates);

% Now replace missing values with predictive distribution
for d=1:ndims
  missing = isnan(X(:,d));
  Nmiss = sum(missing);
  prob = pZ(missing, :) * model.cpd.T(:,:,d); %  p(Xj=v)= sum_k pZ(i,k) T(k,v,j)
  pX(missing, d, :) = reshape(prob, [Nmiss, 1, NobsStates]);
end


return

function logp = mixDiscreteLogprob(model, X)
%% Calculate logp(i) = log p(X(i,:) | model)
% Can handle NaNs
%%

% This file is from pmtk3.googlecode.com

[pZ, logp] = mixDiscreteInferLatent(model, X); 
return

function [pZ, ll] = mixDiscreteInferLatent(model, X)
% Infer latent mixture node from a set of data
% pZ(i, k) = p( Z = k | X(i, :), model) 
% ll(i) = log p(X(i, :) | model)  
% X may contain NaN for missing values (in discrete case)
%%

% This file is from pmtk3.googlecode.com

nmix   = model.nmix; 
[n, d] = size(X); 
logMix = log(rowvec(model.mixWeight)); 
logPz  = zeros(n, nmix); 

logT = log(model.cpd.T + eps);
Lijk = zeros(n, d, nmix);
X = canonizeLabels(X);
for j = 1:d
  ndx = (~isnan(X(:,j)));
  Lijk(ndx, j, :) = logT(:, X(ndx, j), j)'; % T is of size [nstates, nObsStates, d]
end
logPz = bsxfun(@plus, logMix, squeeze(sum(Lijk, 2))); % sum across d

[logPz, ll] = normalizeLogspace(logPz);
pZ          = exp(logPz);
return

function [model, loglikHist, llHists] = emAlgo(model, data, init, estep, mstep, varargin)
% Generic EM algorithm
%
% You must provide the following functions as input
%   model = init(model, data, restartNum) % initialize params
%   [ess, loglik] = estep(model, data) % compute expected suff. stats
%   model = mstep(model, ess) % compute params
%
% Outputs:
% model is a struct returned by the mstep function.
% loglikHist is the history of log-likelihood (plus log-prior) vs
% iteration.  The length of this gives the number of iterations.
% llHists{i} is the history for the i'th random restart;
% loglikHist is the history of the best run
%
% Optional arguments [default]
% maxIter: [100]
% convTol: convergence tolerance [1e-3]
% verbose: [false]
% plotFn: function of form plotfn(model, data, ess, ll, iter), default []
% nRandomRestarts: [1]
% mstepOR: [] this function is required if you use over-relaxed EM

% This file is from pmtk3.googlecode.com


%% Random Restart
[nRandomRestarts, verbose, restartNum, args] = process_options(varargin, ...
    'nrandomRestarts', 1, 'verbose', false, 'restartNum', 1);
if nRandomRestarts > 1
    models  = cell(1, nRandomRestarts);
    llhists = cell(1, nRandomRestarts);
    bestLL  = zeros(1, nRandomRestarts);
    for i=1:nRandomRestarts
        if verbose
            fprintf('\n********** Random Restart %d **********\n', i);
        end
        [models{i}, llhists{i}] = emAlgo(model, data, init, estep,...
            mstep, 'verbose', verbose, 'restartNum', i, args{:});
        bestLL(i) = llhists{i}(end);
    end
    bestndx = maxidx(bestLL);
    model = models{bestndx};
    loglikHist = llhists{bestndx};
    return
end
%% Perform over relaxed EM
[mstepOR, overRelaxFactor, args] = process_options(varargin, 'mstepOR', [], ...
    'overRelaxFactor', []);
if ~isempty(mstepOR)
    [model, loglikHist] = emAlgoAdaptiveOverRelaxed...
        (model, data, init, estep, mstep, mstepOR, args{:});
    return;
end
%% Perform EM
[maxIter, convTol, plotfn, verbose, restartNum] = process_options(args ,...
    'maxIter'    , 100   , ...
    'convTol'    , 1e-4  , ...
    'plotfn'     , []    , ...
    'verbose'    , false , ...
    'restartNum' , 1   );

  if verbose, fprintf('initializing model for EM\n'); end
model = init(model, data, restartNum);
iter = 1;
done = false;
loglikHist = zeros(maxIter + 1, 1);
while ~done
    [ess, ll] = estep(model, data);
    if verbose
        fprintf('%d\t loglik: %g\n', iter, ll );
    end
    if ~isempty(plotfn)
        plotfn(model, data, ess, ll, iter);
    end
    loglikHist(iter) = ll;
    model = mstep(model, ess);
    done  = (iter > maxIter) || ( (iter > 1) && ...
        convergenceTest(loglikHist(iter), loglikHist(iter-1), convTol, true));
    iter = iter + 1;
end
loglikHist = loglikHist(1:iter-1);
llHists{1} = loglikHist;
return

function N = nunique(X, dim)
% Count the unique elements of X along the specified dimension (default 1).
% Like length(unique(X(:, j)) or length(unique(X(i, :)) but vectorized.
% Supports multidimensional arrays, e.g. nunique(X, 3).
%
%
% Example:
%X =
%      5     4     3     3     2     1     4
%      2     1     1     1     1     4     3
%      3     1     4     2     3     4     1
%      2     4     1     1     5     3     1
%      4     5     1     5     1     5     3
%      2     5     4     1     2     2     4
%      2     4     3     3     3     1     4
%      4     1     2     3     1     2     4
%
%
% nunique(X, 1)
% ans =
%      4     3     4     4     4     5     3
%
%
%
% nunique(X, 2)
% ans =
%      5
%      4
%      4
%      5
%      4
%      4
%      4
%      4

if nargin == 1
    dim = find(size(X)~=1, 1);
    if isempty(dim), dim = 1; end
end
N = sum(diff(sort(X, dim), [], dim) > 0, dim) + 1;

return

function Xsplit = randsplit(X, k)
% Split rows of X into k (roughly) equal random partitions
% Xsplit is a k-by-1 cell array. The last cell will have 
% n - (k-1)*floor(n/k) elements, all others will have floor(n/k).

% This file is from pmtk3.googlecode.com


n = size(X, 1);
Xsplit = cell(k, 1);
perm = randperm(n);
psize = floor(n/k);
for i=1:k
    start = psize*(i-1)+1;
    ndx = start:start+psize-1;
    Xsplit{i} = X(perm(ndx), :);
end
if psize*k < n
    Xsplit{end} = [Xsplit{end}; X(perm(psize*k+1:end), :)];
end

return

function model = discreteFit(X, alpha, K)
% Fit a discrete distribution, or if X is a matrix, a product of discrete distributions
%
% X(i, j)        is the ith case, assumed to be from the jth distribution.
%                X must be in 1:K.
%
% alpha        - dirichlet alpha, i.e. pseudo counts
%                (default is all ones vector - i.e. no prior)
%
%
% model        - a struct with the following fields:
%
%     d       - the number of distributions, i.e. size(X, 2)
%     K       - the number of states, i.e. nunique(X)
%     T       - a K-by-d stochastic matrix, (each *column* represents a
%               different distribution).
%
% Example:
% X = randi(5, [100, 4]);   % categorical data in [1,5] 100 cases, 4 dists
% model = discreteFit(X);
%      OR
% alpha = [1 3 3 5 9];
% model = discreteFit(X, alpha);

% This file is from pmtk3.googlecode.com

d = size(X, 2);
X = canonizeLabels(X); % convert to 1..K
if nargin < 3, K  = nunique(X(:)); end
counts = histc(X, 1:K); % works even when X is a matrix - no need to loop
if nargin < 2 || isempty(alpha), alpha = 1; end
model.T = normalize(bsxfun(@plus, counts, colvec(alpha-1)), 1);
model.K = K;
model.d = d;
return

function [canonized, support] = canonizeLabels(labels,support)
%% Transform labels to 1:K
% The size of canonized is the same as labels but every
% label is transformed to its corresponding entry in 1:K. If labels does not
% span the support, specify the support explicitly as the 2nd argument. 
%
% Examples:
%%
% str = {'yes'    'no'    'yes'    'yes'    'maybe'    'no'    'yes'  'maybe'};
%     
% canonizeLabels(str)
% ans =
%      3     2     3     3     1     2     3     1
%%
%canonizeLabels([3,5,8,9; 0,0,-3,2])
%ans =
%     4     5     6     7
%     2     2     1     3
%
%%
% Suppose we know the support is say 10:20 but our labels are [11:15,17,19] and
% we want 11 to be coded as 2 since our support begins at 10 and similarly
% 19 codes as 10 and 20 as 11. We can specify the actual support to achieve
% this.
%
% canonizeLabels([10,11,19,20])          - without specifying support
% ans =  1     2     3     4
%    
% canonizeLabels([10,11,19,20],10:20)        - with specifying support
% ans =  1     2    10    11
% 
%
% To make 0,1 use canonizeLabels(y)-1
% To make -1,+1 use (2*(canonizeLabels(y)-1))-1

% This file is from pmtk3.googlecode.com




[nrows,ncols] = size(labels);
labels = labels(:);

if(nargin == 2)
  labels = [labels;support(:)];
end

if(ischar(labels))
  [s,j,canonized] = unique(labels,'rows');
elseif(issparse(labels))
  labels = double(full(labels));
  [s,j,canonized] = unique(labels);
else
  [s,j,canonized] = unique(labels);
end

if(nargin == 2)
  if(~isequal(support(:),s(:)))
    error('Some of the data lies outside of the support.');
  end
  canonized(end:-1:end-numel(support)+1) = [];
end
support = s;
canonized = reshape(canonized,nrows,ncols);
if ~iscell(labels)
  canonized(isnan(labels))=nan;
end
return

function x = colvec(x)
% Return x as a column vector. This function is useful when a function returns a
% row vector or matrix and you want to immediately reshape it in a functional
% way. Suppose f(a,b,c) returns a row vector, matlab will not let you write
% f(a,b,c)(:) - you would have to first store the result. With this function you
% can write colvec(f(a,b,c)) and be assured that the result is a column vector.
   x = x(:); 
return

function [A, z] = normalize(A, dim)
% Make the entries of a (multidimensional) array sum to 1
% [A, z] = normalize(A) normalize the whole array, where z is the normalizing constant
% [A, z] = normalize(A, dim)
% If dim is specified, we normalize the specified dimension only.
% dim=1 means each column sums to one
% dim=2 means each row sums to one
%
%%
% Set any zeros to one before dividing.
% This is valid, since s=0 iff all A(i)=0, so
% we will get 0/1=0

% This file is from pmtk3.googlecode.com

if(nargin < 2)
    z = sum(A(:));
    z(z==0) = 1;
    A = A./z;
else
    z = sum(A, dim);
    z(z==0) = 1;
    A = bsxfun(@rdivide, A, z);
end
return


function CPD = condDiscreteProdCpdCreate(T, varargin)
%% Create a conditional discrete product distribution
% This differs from a tabularCPD in that it supports vector valued discrete
% observations, (it also supports scalar value observations). 
% These are assumed to be conditionally independent.
%
% T is of size nstates-by-nObsStates-nChildren
% so T(k,j,i) = p(i'th child = j | parent = k)
%
%% Optional inputs
% 'prior' - a struct with the the field 'alpha', which must be
% either a scalar or a matrix the same size as T. 
%%

% This file is from pmtk3.googlecode.com

prior = process_options(varargin, 'prior', []);
if isempty(prior)
   prior.alpha = 1.1; % add 0.1 as pseudo counts (implicitly replicated) 
end
[nstates, nObsStates, d] = size(T); 
CPD            = structure(T, nstates, nObsStates, d, prior);
CPD.cpdType    = 'condDiscreteProd';
CPD.fitFn      = @condDiscreteProdCpdFit;
CPD.fitFnEss   = @condDiscreteProdCpdFitEss;
CPD.essFn      = @condDiscreteProdCpdComputeEss;
CPD.logPriorFn = @(m)sum(log(m.T(:) + eps).*(m.prior.alpha-1));
CPD.rndInitFn  = @rndInit;
return

function CPD = rndInit(CPD)
%% Randomly initialize
CPD.T = normalize(rand(size(CPT.T), 2)); 
return

function CPD = condDiscreteProdCpdFit(CPD, Z, Y)
%% Fit  given fully observed data
% (MAP estimate with Dirichlet prior)
% Z(i) is the state of the parent Z in case i.
% Y(i, :) is the ith 1-by-d observation of the children
%%
nstates = CPD.nstates;
nObsStates = CPD.nObsStates; 
T = CPD.T;
if isempty(CPD.prior)
    alpha = 1;
else
    alpha = CPD.prior.alpha;
end
for k = 1:nstates
   T(k, :, :) = normalize(histc(Y(Z==k, :) + alpha - 1, 1:nObsStates), 2); 
end
return

function ess = condDiscreteProdCpdComputeEss(cpd, data, weights, B)
%% Compute the expected sufficient statistics for a condDiscreteProd CPD
% data     -  nobs-by-d
% weights  -  nobs-by-nstates; the marginal probability of the parent    
% B        -  ignored, but required by the interface, 
%             (since mixture emissions, e.g. condMixGaussTied, use it). 
%%
[nstates, nObsStates, d] = size(cpd.T);
counts  = zeros(nstates, nObsStates, d);% counts(k, c, d) = p(x_d = c | Z = k)
if d < nObsStates*nstates
    for j = 1:d
        counts(:, :, j) = weights'*bsxfun(@eq, data(:, j), 1:nObsStates);
    end
else
    for c = 1:nObsStates
        for k = 1:nstates
            counts(k, c, :) = sum(bsxfun(@times, (data==c), weights(:, k)));
        end
    end
end
ess.counts = counts;
ess.wsum   = sum(weights, 1);
return

function cpd = condDiscreteProdCpdFitEss(cpd, ess)
%% Fit a condDiscreteProdCpd given the expected sufficient statistics
prior = cpd.prior;
if isempty(prior)
    alpha = 1;
else
    alpha = prior.alpha;
end
cpd.T = normalize(ess.counts + alpha-1, 2); 
return


function S = structure(varargin)
% Create a struct directly from variables, without having to provide names
% The current names of the variables are used as the structure fields.
%
% *** does not support anonymous variables as in structure(25, 2+3), etc ***
%
%% Example 
%
% mu = zeros(1, 10);
% Sigma = randpd(10);
% pi = normalize(ones(1, 10)); 
% model = structure(mu, Sigma, pi); 
% model
% model = 
%        mu: [0 0 0 0 0 0 0 0 0 0]
%     Sigma: [10x10 double]
%        pi: [1x10 double]
%%

% This file is from pmtk3.googlecode.com


for i=1:nargin
    S.(inputname(i)) = varargin{i};
end
return

function x = rowvec(x)
% Return x as a row vector. This function is useful when a function returns a
% column vector or matrix and you want to immediately reshape it in a functional
% way. Suppose f(a,b,c) returns a column vector, matlab will not let you write
% f(a,b,c)(:)' - you would have to first store the result. With this function you
% can write rowvec(f(a,b,c)) and be assured that the result is a row vector.   
    x = x(:)';
return

function [y, L] = normalizeLogspace(x)
% Normalize in logspace while avoiding numerical underflow
% Each *row* of x is a log discrete distribution.
% y(i,:) = x(i,:) - logsumexp(x,2) = x(i) - log[sum_c exp(x(i,c)]
% L is the log normalization constant
% eg [logPost, L] = normalizeLogspace(logprior + loglik)
%    post = exp(logPost);
%%

% This file is from pmtk3.googlecode.com

L = logsumexp(x, 2);
%y = x - repmat(L, 1, size(x,2));
y = bsxfun(@minus, x, L);
 
return

function [converged] = convergenceTest(fval, previous_fval, threshold, warn)
% Check if an objective function has converged
%
% We have converged if the slope of the function falls below 'threshold', 
% i.e., |f(t) - f(t-1)| / avg < threshold,
% where avg = (|f(t)| + |f(t-1)|)/2 
% 'threshold' defaults to 1e-4.
% This stopping criterion is from Numerical Recipes in C p423

% This file is from pmtk3.googlecode.com


if nargin < 3, threshold = 1e-4; end
if nargin < 4, warn = false; end

converged = 0;
delta_fval = abs(fval - previous_fval);
avg_fval = (abs(fval) + abs(previous_fval) + eps)/2;
if (delta_fval / avg_fval) < threshold, converged = 1; end

if warn && (fval-previous_fval) < -2*eps %fval < previous_fval
    warning('convergenceTest:fvalDecrease', 'objective decreased!'); 
end

return

function [ynum, xnum] = nsubplots(n)
% Figure out how many plots in the y and x directions to cover n in total
% while keeping the aspect ratio close to rectangular
% but not too stretched

% This file is from pmtk3.googlecode.com


if n==2
  ynum = 2; xnum = 2;
else
  xnum = ceil(sqrt(n));
  ynum = ceil(n/xnum);
end
return

function printPmtkFigure(filename, format, printFolder) %#ok
% print current figure to specified file in .pdf (or other) format

% This file is from pmtk3.googlecode.com

return; % uncomment this to enable printing

if nargin <2, format = 'pdf'; end
if nargin < 3, printFolder = []; end
if isempty(printFolder)
  if ismac
    printFolder = '/Users/kpmurphy/Dropbox/MLbook/Figures/pdfFigures';
  else
    error('need to specify printFolder')
  end
end
if strcmpi(format, 'pdf')
  pdfcrop;
end
fname = sprintf('%s/%s.%s', printFolder, filename, format);
fprintf('printing to %s\n', fname);
if exist(fname,'file'), delete(fname); end % prevent export_fig from appending
if 0
  %opts = struct('Color', 'rgb', 'Resolution', 1200, 'fontsize', 12);
  opts = struct('Color', 'rgb', 'Resolution', 1200);
  exportfig(gcf, fname, opts, 'Format', 'pdf' );
else
  set(gca,'Color','none') % turn off gray background
  set(gcf,'Color','none')
  export_fig(fname)
end
return
