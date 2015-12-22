%This script is written to predict labels on testset, given testdata
%features, and traindata features, instead of SVM, I use KNN.
%there are 3 important params in this script which controls the functions
%of this script
%ifTrain: to decide wheather to run the knn traning part
%ifTest: to decide wheather to run the knn test part
%iforiginalPatches: to decide wheather to update the labels for the patches

function pLabels=predictLables_knn()
addpath('liblinear/');
traindatapath='../results/feature4SVM/allImages/features4SVM_layer1.mat';


ifTrain=0;
ifTest=1;
iforiginalPatches=1;

% if ifTest
% 
% end


load(traindatapath,'layer1pooledstates','trainLabelMat');%note, s is a [width, width,channel,num] form
s=layer1pooledstates;
s=reshape(s,[size(s,1)*size(s,2),size(s,3),size(s,4)]);%Now I want to count fire units for each channel,and use them as feature
layer1channelstates=zeros(size(s,2),size(s,3));%channel*num
for i=1:size(s,3)%num
    for j=1:size(s,2)%channel
        layer1channelstates(j,i)=sum(s(:,j,i));
    end
end
trainBetas=layer1channelstates;

trainedlabel=trainLabelMat;

if ifTrain
    N=size(trainBetas,2);
    rate=0.8;
    N0=floor(N*rate);

    trainindex=randperm(N,N0);
    testindex=setdiff(1:N,trainindex);

    X_train=trainBetas(:,trainindex);
    y_train=trainLabelMat(trainindex);

    X_test=trainBetas(:,testindex);
    y_test=trainLabelMat(testindex);

    K=1;
    y = knn(X_test, X_train, y_train, K);
    cnt=countif(y,y_test);
    tt=y-y_test;
    fprintf('the prediction result when cross validation: %d/%d\n',cnt,length(y));
end

if ifTest
    specTrainFeature_unwhiten=[];
    specTrainFeature_whiten=[];
    specTrainLabel=[];
    for spec=1:17%total 17 species
        spec1=spec;spec2=spec;day=2;
        testdatapath='../results/feature4SVM/maskOneImage/';
        [SVMfeaturefilename,whitenfilename,unwhitenfilename,flag]=getFeatures4OneImage(spec1,spec2,day);

        if flag==1%there exist such spec1,spec2,day
        load([testdatapath,SVMfeaturefilename],'layer1pooledstates','trainLabelMat','specday','rows','cols');
        testLabelMat=trainLabelMat;
        s=layer1pooledstates;
        s=reshape(s,[size(s,1)*size(s,2),size(s,3),size(s,4)]);%Now I want to count fire units for each channel,and use them as feature
        st=s;
        layer1channelstates=zeros(size(s,2),size(s,3));%channel*num
        for i=1:size(s,3)%num
            for j=1:size(s,2)%channel
                layer1channelstates(j,i)=sum(s(:,j,i));
            end
        end
        testBetas=layer1channelstates;
        %testLabelMat=trainedlabel;
        K=1;
        y_estimate=knn(testBetas,trainBetas,trainedlabel,K);
        labelTab=reshape(y_estimate,[cols,rows]);%the order is very important: first cols, then rows
        labelTab=labelTab';
        %Now I want to use some prior information to make the ground prediction
        %for each patch more precise
        labelTab(1,:)=-1;
        labelTab(end,:)=-1;
        labelTab(:,1)=-1;
        labelTab(:,end)=-1;
        for i=2:size(labelTab,1)-1
            for j=2:size(labelTab,2)-1
                if labelTab(i-1,j)==labelTab(i+1,j)&&labelTab(i,j-1)==labelTab(i,j+1)&&labelTab(i-1,j)==labelTab(i,j-1)
                    labelTab(i,j)=labelTab(i-1,j);
                end
            end
        end
        ylabel=reshape(labelTab',1,rows*cols);%after prior info process, and reshpe to [1,rows*cols]
        cnt=countif(y_estimate,testLabelMat);
        tt=y_estimate-testLabelMat;
        fprintf('the prediction result when on one iamge: %d/%d\n',cnt,length(y_estimate));
        d=28;
        recoverImage=repmat(y_estimate,d^2,1);
        recoverImage1=repmat(ylabel,d^2,1);%after prior info process
        display_network_layer1(recoverImage1);
        outpath='../results/cdbnvisual/specinfo/';
        saveas(gcf,[outpath,sprintf('spec%d-%d-day%d-ground.png',spec1,spec2,day)]);
        save([outpath,sprintf('spec%d-%d-day%d-groundinfo.mat',spec1,spec2,day)],'labelTab','ylabel');

    %2.2collect corresponding original patches that are predicted to be foreground
    %patches using unwhitenfilename and ylabel
        if iforiginalPatches
            opatchpath='../../patcheswithlabel/';
            %make the unwhiten patches labeled 
            load([opatchpath,unwhitenfilename],'trainFeatureMat','trainLabelMat','rows','cols','specday');
            if sum(find(trainLabelMat==1))==0%if it hasn't labeled 
                trainLabelMat=ylabel;
            end
            save([opatchpath,unwhitenfilename],'trainFeatureMat','trainLabelMat','rows','cols','specday'); 
            %now I want to collect a set of foreground patches together with their
            %species info 
            fpatches=trainFeatureMat(:,find(trainLabelMat==1));%d*n
            specinfo=specday(:,find(trainLabelMat==1));%spec1,spec2,day:d*n
            specTrainFeature_unwhiten(:,size(specTrainFeature_unwhiten,2)+1:size(specTrainFeature_unwhiten,2)+size(fpatches,2))=fpatches;
            specTrainLabel(:,size(specTrainLabel,2)+1:size(specTrainLabel,2)+size(specinfo,2))=specinfo;
            %make the whiten patches labeled
            load([opatchpath whitenfilename],'trainFeatureMat','trainLabelMat','rows','cols','specday','precomp');
            if sum(find(trainLabelMat==1))==0%if hasn't labeled 
                trainLabelMat=ylabel;
            end
            save([opatchpath whitenfilename],'trainFeatureMat','trainLabelMat','rows','cols','specday','precomp'); 
            fpatches=trainFeatureMat(:,find(trainLabelMat==1));%d*n
            specTrainFeature_whiten(:,size(specTrainFeature_whiten,2)+1:size(specTrainFeature_whiten,2)+size(fpatches,2))=fpatches;

        end
        else%flag==0 which means there is no such image for this species
            fprintf('there is no such image:%d-%d %dd.jpg\n',spec1,spec2,day);
        end%flag==1
    end%for i=1:17
    save([outpath,'specTrainData_step2_unwhiten.mat'],'specTrainFeature_unwhiten','specTrainLabel');
    save([outpath,'specTrainData_step2_whiten.mat'],'specTrainFeature_whiten','specTrainLabel');
end
%now recover the input image using estimated labels

% model = trainll(trainLabelMat',sparse(trainBetas'),'-s 6 -c 1 -v 10','row');
% model = trainll(trainLabelMat',sparse(trainBetas'),'-s 6 -c 1','row');
% %model = trainll(testLabelMat',sparse(testBetas'),'-s 6 -c 1 -v 5','row');
% pLabels = predictll(testLabelMat',sparse(testBetas'),model,'','row');
% %save([testdatapath,'layer3_result_bycdbn5000.mat'],'pLabels');
return

%to count the two number of hit elements
function cnt=countif(x,y)
cnt=0;
if length(x)~=length(y)
    disp('two vectors do not have the same length.');
    return;
end
for i=1:length(x)
    if x(i)==y(i)
        cnt=cnt+1;
    end
end
return