function varargout = gmm(X, K_or_centroids)
% ============================================================
% Expectation-Maximization iteration implementation of
% Gaussian Mixture Model.
%
% PX = GMM(X, K_OR_CENTROIDS)
% [PX MODEL] = GMM(X, K_OR_CENTROIDS)
%
%  - X: N-by-D data matrix.
%  - K_OR_CENTROIDS: either K indicating the number of
%       components or a K-by-D matrix indicating the
%       choosing of the initial K centroids.
%
%  - PX: N-by-K matrix indicating the probability of each
%       component generating each point.
%  - MODEL: a structure containing the parameters for a GMM:
%       MODEL.Miu: a K-by-D matrix.
%       MODEL.Sigma: a D-by-D-by-K matrix.
%       MODEL.Pi: a 1-by-K vector.
% ============================================================
% @SourceCode Author: Pluskid (http://blog.pluskid.org)
% @Appended by : Sophia_qing (http://blog.csdn.net/abcjennifer)
    

%% Generate Initial Centroids
    threshold = 1e-15;
    [N, D] = size(X);
 
    if isscalar(K_or_centroids) %if K_or_centroid is a 1*1 number
        K = K_or_centroids;
        Rn_index = randperm(N); %random index N samples
        centroids = X(Rn_index(1:K), :); %generate K random centroid
    else % K_or_centroid is a initial K centroid
        K = size(K_or_centroids, 1); 
        centroids = K_or_centroids;
    end
 
    %% initial values
    [pMiu pPi pSigma] = init_params();
 
    Lprev = -inf; %上一次聚类的误差
    
    %% EM Algorithm
    while true
        %% Estimation Step
        Px = calc_prob();
 
        % new value for pGamma(N*k), pGamma(i,k) = Xi由第k个Gaussian生成的概率
        % 或者说xi中有pGamma(i,k)是由第k个Gaussian生成的
        pGamma = Px .* repmat(pPi, N, 1); %分子 = pi(k) * N(xi | pMiu(k), pSigma(k))
        pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K); %分母 = pi(j) * N(xi | pMiu(j), pSigma(j))对所有j求和
 
        %% Maximization Step - through Maximize likelihood Estimation
        
        Nk = sum(pGamma, 1); %Nk(1*k) = 第k个高斯生成每个样本的概率的和，所有Nk的总和为N。
        
        % update pMiu
        pMiu = diag(1./Nk) * pGamma' * X; %update pMiu through MLE(通过令导数 = 0得到)
        pPi = Nk/N;
        
        % update k个 pSigma
        for kk = 1:K 
            Xshift = X-repmat(pMiu(kk, :), N, 1);
            pSigma(:, :, kk) = (Xshift' * ...
                (diag(pGamma(:, kk)) * Xshift)) / Nk(kk);
        end
 
        % check for convergence
        L = sum(log(Px*pPi'));
        if L-Lprev < threshold
            break;
        end
        Lprev = L;
    end
 
    if nargout == 1
        varargout = {Px};
    else
        model = [];
        model.Miu = pMiu;
        model.Sigma = pSigma;
        model.Pi = pPi;
        varargout = {Px, model};
    end
 
    %% Function Definition
    
    function [pMiu pPi pSigma] = init_params()
        pMiu = centroids; %k*D, 即k类的中心点
        pPi = zeros(1, K); %k类GMM所占权重（influence factor）
        pSigma = zeros(D, D, K); %k类GMM的协方差矩阵，每个是D*D的
 
        % 距离矩阵，计算N*K的矩阵（x-pMiu）^2 = x^2+pMiu^2-2*x*Miu
        distmat = repmat(sum(X.*X, 2), 1, K) + ... %x^2, N*1的矩阵replicateK列
            repmat(sum(pMiu.*pMiu, 2)', N, 1) - ...%pMiu^2，1*K的矩阵replicateN行
            2*X*pMiu';
        [~, labels] = min(distmat, [], 2);%Return the minimum from each row
 
        for k=1:K
            Xk = X(labels == k, :);
            pPi(k) = size(Xk, 1)/N;
            pSigma(:, :, k) = cov(Xk);
        end
    end
 
    function Px = calc_prob() 
        %Gaussian posterior probability 
        %N(x|pMiu,pSigma) = 1/((2pi)^(D/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pMiu)'pSigma^(-1)*(x-pMiu))
        Px = zeros(N, K);
        for k = 1:K
            Xshift = X-repmat(pMiu(k, :), N, 1); %X-pMiu
            inv_pSigma = inv(pSigma(:, :, k));
            tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);
            coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma));
            Px(:, k) = coef * exp(-0.5*tmp);
        end
    end
end
