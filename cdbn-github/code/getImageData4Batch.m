%获取一个文件夹下的所有.jpg对应的patch
function imdata_batch=getImageData4Batch(fpath,batch_size,filtersize,onepatchrow,onepatchcol)
%y = randsample(n,k,replacement) or y = randsample(population,k,replacement) returns a sample 
    %taken with replacement if replacement is true, or without replacement if replacement is false. The default is false.
    images_all=sample_images_all(fpath);
    for i = 1:length(images_all)%对每张图片抽取batch_size张batch_ws大小的图片(就是patch大小)
        imdata = images_all{i};
        rows = size(imdata,1);
        cols = size(imdata,2);

        for batch=1:batch_size%上面的一个imbatch,是一张图片，将一张图片subsample出3张70*70的patch,这里每个batch是一个70*70的patch
            rowidx = ceil(rand*(rows-2*filtersize-onepatchrow))+filtersize + [1:onepatchrow];%ws是卷积核大小
            colidx = ceil(rand*(cols-2*filtersize-onepatchcol))+filtersize + [1:onepatchcol];
            k=3*(i-1)+batch;%一张图片取三个patch
            imdata_batch(:,:,k) = imdata(rowidx, colidx);%从imdata（一张处理后的图片)取子patch
            patch_k=imdata_batch(:,:,k);
            imdata_batch(:,:,k) = imdata_batch(:,:,k) - mean(patch_k(:));%减去均值
        end
    end
end