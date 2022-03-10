close all;
startup;

sizes = [1000, 10000, 100000, 1000000, 10000000, 50000000]
%sizes = [5000000]
kwant = 5
repeat = 1;
what = "abs"

% parameters for Chebyshev-Davidson method
m = 11;
tau = 1e-3
itermax = 2000
%opts = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1);

% parameters for Coordinate-wise Descent
stepsize = 0.02
w = 1.0
alpha = 0.9
th = 0.3;
a0 = 0.1;
p = 1.5;
opts_blk = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'kmore', 0, 'blk', kwant, 'upb', 2.1+a0);

%ncores = 40
%parpool(ncores, 'IdleTimeout', 360);

batch = 50;
fnorm = 'fro';
for n_samples = sizes
    % n_samples = 5000
    fprintf("\n\n")
    fprintf("========================= #samples = %10d ============================\n", n_samples)
    fname = "sparsedata/sparse" + num2str(n_samples) + "/sparse" + num2str(n_samples) + what + ".mat"; 
    %fid = fopen(fname, 'r'); 
    %raw = fread(fid, inf); 
    %str = char(raw');     
    
    %A = sparseSim(str, n_samples, what);
    A = load(fname);
    A = A.A;
    
    A = (A+A')/2;

    itermaxCoD = itermax*n_samples;

    if strcmp(what, "pos")
        %A = posMat(A);
        %D = constructD(A);
        D = sparse([1:n_samples], [1:n_samples], sum(A), n_samples, n_samples);
        L = constructL(D, A);
    elseif strcmp(what, "bin")
        %A = binMat(A);
        %D = constructD(A);
        D = sparse([1:n_samples], [1:n_samples], sum(A), n_samples, n_samples);
        L = constructL(D, A);
    elseif strcmp(what, "abs")
        %D = constructD_abs(A);
        D = sparse([1:n_samples], [1:n_samples], sum(abs(A)), n_samples, n_samples);
        L = constructL(D, A);
    end

    dL = sparse([1:n_samples], [1:n_samples], a0*ones(n_samples,1), n_samples, n_samples);
    L = L + dL;
    L = (L + L')/2;
    
    nonzerocols = findnnz(L);

    %loss = @(c)sum(sum((L+c*c').^2));

    V0 = randn(n_samples, kwant);
    fprintf("nonzeros rate: %10.4f \n", nnz(L)/n_samples/n_samples)
    %fprintf("initial loss: %10.4e \n\n", loss(V0))
    
    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = bchdav(L, kwant, opts_blk);
    end
    time = toc/repeat;


    %fprintf("--------------------------------------------\n")
    fprintf("results from bchdav: \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    fprintf("--------------------------------------------\n")    


    tic;
    for i = 1:repeat
        [eigV, eigW] = eigs(L, kwant, 'smallestabs', 'Tolerance', tau);
    end
    timeE = toc/repeat;

    %fprintf("--------------------------------------------\n")
    fprintf("results from eigs: \n")
    fprintf("running time %.4f \n", timeE)
    fprintf("computed eigenvalues: %f \n", sort(diag(eigW))-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigV - eigV*eigW, fnorm)/norm(eigV*eigW, fnorm), loss(eigV))
    fprintf("eigV error: %10.4e \n", norm(L*eigV - eigV*eigW, 'fro')/norm(eigV*eigW, 'fro'))
end

exit
