%This script is written to run the example of my cdbn model, feature
%learning process, svm train and test process

cdbnpath='../results/cdbnmodel/';
load(cdbnfile,'crbm');
par1=paramsetup();
save(cdbnfile,'crbm','par1');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%learn features%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %datapath='../patcheswithlabel/data-19-Nov-2014Gray_2020_pure_unwhiten.mat';
 %[cdbn,svmmodel]=doWholeProcess(datapath,cdbnfile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%train models%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%datapath='../../patcheswithlabel/mnist_uint8.mat';
%datapath='../patcheswithlabel/data-19-Nov-2014Gray_2020_pure_whiten.mat';
%datapath='../patcheswithlabel/data-20-Nov-2014_whiten_pure_rgb2020.mat';
%datapath='../../patcheswithlabel/data-20-Nov-2014Gray_5050_pure_whiten.mat';
%datapath='../../patcheswithlabel/data-28-Jan-2015Edge2828_sub4_image1_20_pure_whiten.mat';
%datapath='../../patcheswithlabel/data-05-Feb-2015Edge2828_sub4_image1_1000_pure_isfore1_isback0_whiten.mat';
%datapath='../../patcheswithlabel/data-11-Feb-2015Gray2828_sub4_image1_158_pure_whiten.mat';%extract test images 
datapath='../../patcheswithlabel/data-11-Feb-2015Gray2828_sub4_step14_image1_158_pure_whiten.mat';
[cdbn,svmmodel]=doWholeProcess(datapath);
