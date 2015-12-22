%This script is written to transform labels when it is predicted 0 to 1 or
%-1
function [ output_args ] = transformLabels( input_args )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fpath='..\results\maskImage\feature4 1-1-2 d databycdbn1000\colwisepatches_givencode\';
filename=[fpath,'data-coverOneImage29-Oct-2014 unwhiten.mat'];
load(filename,'labels');
fid=fopen([fpath,'result_ok'],'w');
labels=squeeze(labels);
for i=1:length(labels)
    fprintf(fid,'%d\n',labels(i));
end
fclose(fid);
end

