%这个函数是求p(h|v)*vi*hj,由于求dP/dW是由两个部分组成sum(p(h|v0)*dE/dW)-sum(p(h|vk)*dE/dW),本函数就是求其中posive部分
%I rewrite this function to make it fit for batchsize-examples at a time,
%and improve the computation performance, also, I make it fit for
%multi-channel visible data.
%Params:
%visdata:提供vi
%hiddata：提供p(h|v),即hj
%filtersize:卷积核大小
%params
%visdata:[patchsize,patchsize,numchannels,batchsize]
%hiddata:hidden layer unit's probabilities p(h|v):[hidpatchsize,hidpatchsize, numfilters,batchsize]
%output
%vishidprod2:[filtersize,filtersize,numchannels,numfilters]
%Date:11/22/2014
%by: Dong Nie
function vishidprod2 = crbm_vishidprod_fixconv(visdata, hiddata, filtersize)

numchannels = size(visdata,3);
numfilters = size(hiddata,3);
batchsize=size(visdata,4);

% tic
% TODO: single channel version is not implemented yet.. Might need to
% modify mexglx file
% selidx1 = size(hiddata,1):-1:1;
% selidx2 = size(hiddata,2):-1:1;
% vishidprod2 = zeros(filtersize,filtersize,numchannels,numfilters);
% 
% %This just compute example by example, it is not very fast
% tic;
% if numchannels==1
%     vishidprod2=squeeze(vishidprod2);
%     for i=1:batchsize
%         vishidprod2 = vishidprod2+conv2_mult(visdata(:,:,:,i), hiddata(selidx1, selidx2, :,i), 'valid');%将imdata(就是输入数据)和隐层数据做卷积,实际上就是product:vi*p(hj|v)
%     end
%     %vishidprod2 = reshape(vishidprod2,[filtersize,filtersize,numchannels,numfilters]);
% else  
%     for b=1:numfilters
%         for c=1:size(visdata,3)%loop for channels
%              for i=1:batchsize
%                 vishidprod2(:,:,c,b) = vishidprod2(:,:,c,b) + conv2(visdata(:,:,c,i), hiddata(selidx1, selidx2, b,i), 'valid');%in fact, conv2_mult is equal to conv2 here
%              end
%         end
%         %原来的语句没有上面的for c=1:..循环
%         %vishidprod2(:,:,:,b) = conv2_mult(visdata, hiddata(selidx1, selidx2, b), 'valid');
%     end
% end
% toc;
% vishidprod2 = reshape(vishidprod2, [filtersize^2, numchannels, numfilters]);
% v1=vishidprod2;

%fprintf('now convn begins in visprod function\n');

selidx1 = size(hiddata,1):-1:1;
selidx2 = size(hiddata,2):-1:1;
vishidprod2 = zeros(filtersize,filtersize,numchannels,numfilters);
%I try to compute the vishidproduct considering multiple examples at a time
%by using convn
%tic;
if numchannels==1
    vishidprod2=squeeze(vishidprod2);
    for i=1:batchsize
        vishidprod2 = vishidprod2+conv2_mult(visdata(:,:,:,i), hiddata(selidx1, selidx2, :,i), 'valid');%将imdata(就是输入数据)和隐层数据做卷积,实际上就是product:vi*p(hj|v)
    end
else
    for k=1:numfilters
        %for c=1:size(visdata,3)%loop for channels
        for i=1:batchsize
            %vishidprod2(:,:,c,b) = vishidprod2(:,:,c,b) + conv2_mult(visdata(:,:,c), hiddata(selidx1, selidx2, b), 'valid');
            vishidprod2(:,:,:,k) = vishidprod2(:,:,:,k) + convn(visdata(:,:,:,i), hiddata(selidx1, selidx2, k,i), 'valid');
        end
        %end
        %原来的语句没有上面的for c=1:..循环
        %vishidprod2(:,:,:,b) = conv2_mult(visdata, hiddata(selidx1, selidx2, b), 'valid');
    end
end
%toc;
vishidprod2 = reshape(vishidprod2, [filtersize^2, numchannels, numfilters]);
%v2=vishidprod2;

return