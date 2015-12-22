function [crbm] =crbmtrain(x,crbm,pars)%应该要一个crbm参数，否则哪里知道当前层的一些参数
%Input:
%x is a 4-d matrix:[row,col,channel,num],when it is frist layer, channel=1
%crbm contains crbm's W, b,c of current layer
%pars contains all necessary parameters
%Output:
%crbm model for this layer

if mod(pars.filtersize,2)~=0, 
    error('filtersize must be even number'); 
end


C_sigm = 1;


numchannels=size(x,3);

error_history = [];
sparsity_history = [];


numofpatches=size(x,4);
numbatches=ceil(numofpatches/pars.batchsize);
%note, as I have randperm data in traincdbnmodel.m,so I donot need to
%randperm here.
 for iter=1:pars.numepochs
     tic;
     ferr_current_iter = [];
     sparsity_curr_iter = [];
     if iter<5,
         pars.momentum = pars.initialmomentum;%0.5
     else
         pars.momentum = pars.finalmomentum;%0.9
     end
     for i=1:numbatches
        imdata_batch=x(:,:,:,(i-1)*pars.batchsize+1:min(i*pars.batchsize,numofpatches));
        imdata_batch = trim_image_for_spacing_fixconv(imdata_batch, crbm.filtersize, pars.spacing);%trimImage

%             if rand()>0.5,
%                 imdata_batch = fliplr(imdata_batch);%fliplr函数实现矩阵左右翻转
%             end

        % use current batch data to update update network
        [ferr dW dh dv poshidprobs poshidstates negdata]= crbm_update_networks(imdata_batch, crbm.W, crbm.hbias_vec, crbm.vbias_vec, pars);
        ferr_current_iter = [ferr_current_iter, ferr];
        sparsity_curr_iter = [sparsity_curr_iter, mean(poshidprobs(:))];%用hide层p(h=1|v)的均值来表示sparsity

         
        
        % update parameters
        crbm.Winc = pars.momentum*crbm.Winc + pars.learningRate*dW;%dw=momentum*dw_old+learningRate*dw
        crbm.W = crbm.W + crbm.Winc;%here I don't implement weight decay yet

        crbm.vbiasinc = pars.momentum*crbm.vbiasinc + pars.learningRate*dv;%db..
        crbm.vbias_vec = crbm.vbias_vec + crbm.vbiasinc;

        crbm.hbiasinc = pars.momentum*crbm.hbiasinc + pars.learningRate*dh;%dc..
        crbm.hbias_vec = crbm.hbias_vec + crbm.hbiasinc;
     end
        
        if (pars.std_gaussian > pars.sigma_stop) % stop decaying after some point( till pars.sigma_stop)
            pars.std_gaussian = pars.std_gaussian*0.99;
         end
     
        if (pars.learningRate>pars.lRateStop)
            pars.learningRate=pars.learningRate*1/(1+0.01*iter);
        end
        mean_err = mean(ferr_current_iter);
        mean_sparsity = mean(sparsity_curr_iter);
       
        error_history(iter) = mean(ferr_current_iter);
        sparsity_history(iter) = mean(sparsity_curr_iter);
        if mod(iter,100)==0
             W1=crbm.W;
            save(sprintf('../results/cdbnmodel/crbmmodel_learningRate%g_l2reg%g_pbias%g_pbias_lambda%g_sigmastart%g_sigmastop%g_bytraindata%d_%s_layer%d_epoch%d.mat',pars.learningRate,pars.l2reg,pars.pbias,pars.pbias_lambda,pars.sigma_start,pars.sigma_stop,size(x,4),date,pars.currentLayer,iter),'crbm');
            figure(1),display_network_layer1(W1);
            saveas(gcf,sprintf('../results/cdbnvisual/learningRate%gl2reg%gpbias%gpbiaslamada%gsigmastart%gsigmastop%g%slayer%d_epoch_%d.png',pars.learningRate,pars.l2reg,pars.pbias,pars.pbias_lambda,pars.sigma_start,pars.sigma_stop,date,pars.currentLayer,iter));  
        end
      fprintf('||W||=%g, ||dWprod|| = %g\n', sqrt(sum(crbm.W(:).^2)), sqrt(sum(dW(:).^2)));
      fprintf('epoch %d error = %g \tsparsity_hid = %g\n', iter, mean(ferr_current_iter), mean(sparsity_curr_iter));
      toc;
      fprintf('The time for a epoch is\n');
 end
 return