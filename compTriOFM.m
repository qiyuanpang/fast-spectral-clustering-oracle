close all; clear all; clc;
startup;

n_samples = 1000
fname = num2str(n_samples) + "/sim" + num2str(n_samples) + ".json";
fid = fopen(fname, 'r');
raw = fread(fid, inf);
str = char(raw');
A = jsondecode(str);

what = "pos"

if strcmp(what, "pos")
    A = posMat(A);
    D = constructD(A);
    L = constructL(D, A);
elseif strcmp(what, "bin")
    A = binMat(A);
    D = constructD(A);
    L = constructL(D, A);
elseif strcmp(what, "abs")
    D = constructD_abs(A);
    L = constructL(D, A);
end

L = L + 0.1*eye(n_samples);

kwant = 2
tau = 1e-8;

% parameters for Coordinate-wise Descent
stepsize = 0.001;
itermaxCoD = 10000;
nonzerocols = {};
for i = 1:n_samples
    nonzerosidx = [];
    for j = 1:n_samples
        if abs(L(i,j)) > 1E-14
            nonzerosidx(end+1) = j;
        end
    end
    nonzerocols{end+1} = nonzerosidx;
end

loss = @(c)sum(sum((L+c*c').^2));

V0 = randn(n_samples, kwant);
loss(V0)


tic;
for i = 1:5
    %[eigenvalues, eigenvectors, nconv, history] = chdav(L, kwant, opts);
    %[eigenvalues, eigenvectors] = CoordinateDescent(L, kwant, stepsize, nonzerocols, itermaxCoD, V0);
    [eigenvalues, eigenvectors] = triofm(L, kwant, stepsize, itermaxCoD);
end
time = toc/5;

fprintf("modified TriOFM: \n")
time
eigenvalues
%eigenvectors
norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues))
loss(eigenvectors)

tic;
for i = 1:5
    %[eigenvalues, eigenvectors, nconv, history] = chdav(L, kwant, opts);
    %[eigenvalues, eigenvectors] = CoordinateDescent(L, kwant, stepsize, nonzerocols, itermaxCoD, V0);
    [eigenvalues, eigenvectors] = triofm_org(L, kwant, stepsize, itermaxCoD);
end
time = toc/5;

fprintf("original TriOFM: \n")
time
eigenvalues
%eigenvectors
norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues))
loss(eigenvectors)

tic;
for i = 1:5
    [eigV, eigW] = eigs(L, kwant, 'smallestabs', 'Tolerance', tau);
end
timeE = toc/5;

fprintf("builtin eigs: \n")
timeE
diag(eigW)
norm(L*eigV - eigV*eigW)/norm(eigV*eigW)
loss(eigV)

exit
