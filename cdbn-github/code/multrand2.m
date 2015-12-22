function [S P] = multrand2(P)
% P is 2-d matrix: 2nd dimension is # of choices

% sumP = row_sum(P); 
tempP=P;
for i=1:size(P,1)
    maxi=max(P(i,:));
    P(i,:)=P(i,:)-maxi;
end
P=exp(P);
tempP1=P;
% if (sum(sum(isnan(P)))>=1)
%     fprintf('P in multrand2 function have %g NaN value\n',sum(sum(isnan(P))));
%     P
% end
sumP = sum(P,2);
P = P./repmat(sumP, [1,size(P,2)]);%终于找到出问题的地方了，原来是这里
%最终，确定了，是因为P里有Inf，sumP里也有Inf(可能更多),当Inf/Inf时，会出现NaN
%所以，问题变成了P为什么会出现Inf呢

% if (sum(sum(isnan(P)))>=1)
%     fprintf('P in multrand2 function have %g NaN value\n',sum(sum(isnan(P))));
%     P
% end

%下面是Bernoulli化的过程
cumP = cumsum(P,2);
% rand(size(P));
unifrnd = rand(size(P,1),1);
temp = cumP > repmat(unifrnd,[1,size(P,2)]);
Sindx = diff(temp,1,2);%按列求一阶近似微分(实际就是相邻两列做差)
S = zeros(size(P));
S(:,1) = 1-sum(Sindx,2);
S(:,2:end) = Sindx;

end
