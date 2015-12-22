%This script is written to show what the patchset looks like.
%params
%patchset:[d,n]
function displayInputPatchset(patchset)
[dim,num]=size(patchset);
if floor(sqrt(num))^2 ~= num%numberoffilters, compute the show figure width and height(m,n)
    n=ceil(sqrt(num));
    while mod(num, n)~=0 && n<1.2*sqrt(num), n=n+1; end
    m=ceil(num/n);
else
    n=sqrt(num);
    m=n;
end
graywidth=sqrt(dim);
rgbwidth=sqrt(dim/3);
channel=1;
if graywidth==floor(graywidth)
    width=graywidth;
elseif rgbwidth==floor(rgbwidth)
    width=rgbwidth;
    channel=3;
else
    fprintf('input data has an error\n');
end
buf=1;
array=ones(buf+m*(width+buf),buf+n*(width+buf),channel);%This is used to 

k=1;
for i=1:m
    for j=1:n
        if k>num, 
            continue; 
        end
        array(buf+(i-1)*(width+buf)+(1:width),buf+(j-1)*(width+buf)+(1:width),:)=reshape(patchset(:,k),[width,width,channel]);
%         for c=1:channel
%             clim=max(abs(A(:,c,k)));
%             if opt_normalize
%                 array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz),c)=reshape(A(:,c,k),sz,sz)/clim;
%             else
%                 array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz),c)=reshape(A(:,c,k),sz,sz)/max(abs(A(:)));
%             end
%         end
         k=k+1;
    end
end
mat=normalizeMatrix(array);
mat=(array);
imshow(mat);
saveas(gcf,sprintf('../results/cdbnvisual/patchset_channel%d_datasize%d_date%s.png',channel,num,date));
end