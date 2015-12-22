%i visualize the 2-layer weights w2 as a weighted linear combination of the first layer bases w1 by the following code:

function [combine_W2]=visualise_2layerweights(W2,W1) 
ws2 = sqrt(size(W2,1)); 
numbases2 = size(W2,3); 
numchannel2 = size(W2,2); %numbases1

ws1 = sqrt(size(W1,1)); 
numbases1 = size(W1,3); 
numchannel1 = size(W1,2); %1

combine_W2 = zeros((ws2+ws1-1)^2,numchannel1, numbases2);

for c=1:numbases2 
    W2_temp=zeros(ws2+ws1-1, ws2+ws1-1,numchannel1); 
    W2_temp1=zeros((ws2+ws1-1)^2,numchannel1);
    for b = 1:numbases1 
        H1 = reshape(W1(:,:,b),[ws1,ws1,numchannel1]); 
        H2 = reshape(W2(:,b,c),[ws2,ws2]); 
        W2_temp = W2_temp + conv2_mult(H1, H2, 'full');
    end
    W2_temp1= reshape(W2_temp, [(ws2+ws1-1)^2,numchannel1]); 
    combine_W2(:,numchannel1,c)=W2_temp1;
end
return

function y = conv2_mult(B, a, convopt)
y = zeros(sqrt(size(B,1))+sqrt(size(a,1))-1,sqrt(size(B,1))+sqrt(size(a,1))-1);

for ii=1:size(a,3)
    b1=reshape(squeeze(B(:,ii)),sqrt(size(B,1)),sqrt(size(B,1)));
    a1=reshape(squeeze(a(:,:,ii)),sqrt(size(a,1)),sqrt(size(a,1)));
    y = y + conv2(b1, a1, convopt);
end
return


% function y = conv2_mult(a, B, convopt)
% y = [];
% for i=1:size(B,3)
%     y(:,:,i) = conv2(a, B(:,:,i), convopt);
% end
% return