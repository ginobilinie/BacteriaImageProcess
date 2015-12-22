function y = knn(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, P-by-N_test matrix.
%           X_train:   training sample features, P-by-N matrix.
%           y_train:   training sample labels, 1-by-N row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1-by-N_test row vector.
%
% Author: Ren Kan
[~,N_test] = size(X);

predicted_label = zeros(1,N_test);
for i=1:N_test
    [dists, neighbors] = top_K_neighbors(X_train,y_train,X(:,i),K); 
    % calculate the K nearest neighbors and the distances.
    predicted_label(i) = recog(y_train(neighbors),length(unique(y_train)));
    % recognize the label of the test vector.
end

y = predicted_label;

return

function [dists,neighbors] = top_K_neighbors( X_train,y_train,X_test,K )
% Author: Dong Nie
%   Input: 
%   X_test the test vector with P*1
%   X_train and y_train are the train data set
%   K is the K neighbor parameter
%to avoid 0 vector, I add 1 to each element, and in fact, this doesn't
%affact the order
X_train=X_train+1;
X_test=X_test+1;

[~, N_train] = size(X_train);
test_mat = repmat(X_test,1,N_train);

% The distance is the Euclid Distance.
% dist_mat = (X_train-double(test_mat)) .^2;
% dist_array = sum(dist_mat);

%Here I choose cosine
norm_test=norm(X_test);
for i=1:N_train
    dist_array(i)=1-X_test'*X_train(:,i)/(norm_test*norm(X_train(:,i)));
end

[dists, neighbors] = sort(dist_array);
% The neighbors are the index of top K nearest points.
dists = dists(1:K);
neighbors = neighbors(1:K);

return

function result = recog( K_labels,class_num )
%RECOG Summary of this function goes here
%   Author: Dong Nie
[~,K] = size(K_labels);
class_count = zeros(1,class_num+1);
for i=1:K
    class_index = K_labels(i)+2; % +1 is to avoid the 0 index reference.
    class_count(class_index) = class_count(class_index) + 1;
end
[~,result] = max(class_count);
result = result - 2; % Do not forget -1 !!!

return