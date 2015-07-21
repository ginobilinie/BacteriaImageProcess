%this script is written to test gpuArray, gpuArray('methods');
%after test, I found using gpuArray, it can speedup to 2-15 times on my own
%laptop.
%1/28/2015
%Dong Nie
function testgpuArray()
testGPU_conv();
return

function testGPU_xcorr()
fprintf('Benchmarking GPU-accelerated Cross-Correlation.\n');

if ~(parallel.gpu.GPUDevice.isAvailable)
    fprintf(['\n\t**GPU does not have a compute capability of 1.3 or ' ...
             'greater. Stopping.**\n']);
    return;
else
    dev = gpuDevice;
    fprintf(...
    'GPU detected (%s, %d multiprocessors, Compute Capability %s\n)',...
    dev.Name, dev.MultiprocessorCount, dev.ComputeCapability);
end

fprintf('\n\n *** Benchmarking vector-vector cross-correlation*** \n\n');
fprintf('Benchmarking function :\n');
type('benchXcorrVec');
fprintf('\n\n');

sizes = [2000 1e4 1e5 5e5 1e6];
tc = zeros(1,numel(sizes));
tg = zeros(1,numel(sizes));
numruns = 10;

for s=1:numel(sizes);
    fprintf('Running xcorr of %d elements...\n', sizes(s));
    delchar = repmat('\b', 1,numruns);

    a = rand(sizes(s),1);
    b = rand(sizes(s),1);
    tc(s) = benchXcorrVec(a, b, numruns);
    fprintf([delchar '\t\tCPU  time : %.2f ms\n'], 1000*tc(s));
    tg(s) = benchXcorrVec(gpuArray(a), gpuArray(b), numruns);
    fprintf([delchar '\t\tGPU time :  %.2f ms\n'], 1000*tg(s));
end

%Plot the results
fig = figure;
ax = axes('parent', fig);
semilogx(ax, sizes, tc./tg, 'r*-');
ylabel(ax, 'Speedup');
xlabel(ax, 'Vector size');
title(ax, 'GPU Acceleration of XCORR');
drawnow;

return

function testGPU_convn()
m=1000;
n=100;
k=5;

gc=convn(gpuArray.rand(m,m,10,'single'),gpuArray.rand(k,'single'));

tic;
for i=1:n
    gc=convn(gpuArray.rand(m,m,10,'single'),gpuArray.rand(k,'single'));
end
toc

c=convn(rand(m,m,10,'single'),rand(k,'single'));
tic;
for i=1:n
    c=convn(rand(m,m,10,'single'),rand(k,'single'));
end
toc
return

function testGPU_conv()
a=randn(2500,2500);
b=randn(100,100);
tic;
c0=conv2(a,b,'valid');
toc;
fprintf('test time for using gpu\n');
tic;
ag=gpuArray(a);
bg=gpuArray(b);
cg=conv2(ag,bg,'valid');
c=gather(cg);
toc;
return
