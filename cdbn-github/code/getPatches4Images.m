%获取一个文件夹下的所有.jpg对应的patches,从每个jpg文件subsample出batch_size个[onepatchrow,onepatchcol]大小的patch
%所以这里返回结果是个四个维度的数据:[onepatchrow,onepatchcol,batch_size,imagenumbersl
function imdata_batch=getPatches4Images(fpath,batch_size,filtersize,onepatchrow,onepatchcol)
%y = randsample(n,k,replacement) or y = randsample(population,k,replacement) returns a sample 
    %taken with replacement if replacement is true, or without replacement if replacement is false. The default is false.
    images_all=sample_images_all(fpath);
    for i = 1:length(images_all)%对每张图片抽取batch_size张batch_ws大小的图片(就是patch大小)
        imdata = images_all{i};
        rows = size(imdata,1);
        cols = size(imdata,2);
        
        for batch=1:batch_size%上面的一个imbatch,是一张图片，将一张图片subsample出batch_size张70*70的patch,这里每个batch是一个70*70的patch
            rowidx = ceil(rand*(rows-2*filtersize-onepatchrow))+filtersize + [1:onepatchrow];%ws是卷积核大小
            colidx = ceil(rand*(cols-2*filtersize-onepatchcol))+filtersize + [1:onepatchcol];
            batchdata(:,:,batch) = imdata(rowidx, colidx);%从imdata（一张处理后的图片)取子patch
            patch_k=batchdata(:,:,batch);
            batchdata(:,:,batch) = batchdata(:,:,batch) - mean(patch_k(:));%减去均值
        end
        imdata_batch(:,:,:,i)=batchdata;
    end
end