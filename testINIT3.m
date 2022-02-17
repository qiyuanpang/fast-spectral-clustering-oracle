close all;
startup;

SIZE = 5000000;
sizes = [100000]
%sizes = [1000]
kwant = 5
repeat = 1;
what = "bin"

% parameters for Chebyshev-Davidson method
m = 11;
tau = 1e-4
itermax = 3000



% parameters for Coordinate-wise Descent
stepsize = 0.02
w = 1.0
alpha = 0.9
th = 0.3;
a0 = 0.1;
p = 1.5;

%ncores = 40
%parpool(ncores, 'IdleTimeout', 360);

batch = 50;
fnorm = 'fro';

inits = 2    
add = 50;
%fprintf("\n\n")
%fprintf("========================= #samples = %10d ============================\n", n_samples)
fname = "sparsedata/" + num2str(SIZE) + "/sparse" + num2str(SIZE) + what + ".mat"; 
A = load(fname);
A = A.A;
    
A = (A+A')/2;

opts = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1, 'upb', 3+a0);
opts_blk = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'kmore', 0, 'blk', kwant, 'upb', 3+a0);

    
for n_samples = sizes
    
    fprintf("\n\n")
    fprintf("========================= #samples = %10d ============================\n", n_samples)
    
    
    n_sample1 = n_samples;
    A1 = A(1:n_sample1,1:n_sample1);

    if strcmp(what, "pos")
        D1 = sparse([1:n_sample1], [1:n_sample1], sum(A1), n_sample1, n_sample1);
        L1 = constructL(D1, A1);
    elseif strcmp(what, "bin")
        D1 = sparse([1:n_sample1], [1:n_sample1], sum(A1), n_sample1, n_sample1);
        L1 = constructL(D1, A1);
    elseif strcmp(what, "abs")
        D1 = sparse([1:n_sample1], [1:n_sample1], sum(abs(A1)), n_sample1, n_sample1);
        L1 = constructL(D1, A1);
    end

    dL = sparse([1:n_sample1], [1:n_sample1], a0*ones(n_sample1,1), n_sample1, n_sample1);
    L1 = L1 + dL;
    L1 = (L1 + L1')/2;

    n_sample2 = n_samples + add;
    A2 = A(1:n_sample2,1:n_sample2);

    if strcmp(what, "pos")
        D2 = sparse([1:n_sample2], [1:n_sample2], sum(A2), n_sample2, n_sample2);
        L2 = constructL(D2, A2);
    elseif strcmp(what, "bin")
        D2 = sparse([1:n_sample2], [1:n_sample2], sum(A2), n_sample2, n_sample2);
        L2 = constructL(D2, A2);
    elseif strcmp(what, "abs")
        D2 = sparse([1:n_sample2], [1:n_sample2], sum(abs(A2)), n_sample2, n_sample2);
        L2 = constructL(D2, A2);
    end

    dL = sparse([1:n_sample2], [1:n_sample2], a0*ones(n_sample2,1), n_sample2, n_sample2);
    L2 = L2 + dL;
    L2 = (L2 + L2')/2;
    
    if inits == 1
        dL = L2(n_sample1+1:end, n_sample1+1:end);
    elseif inits == 2
        dA = A2(n_sample1+1:end, n_sample1+1:end);
        if strcmp(what, "pos")
            dD = sparse([1:n_sample2-n_sample1], [1:n_sample2-n_sample1], sum(dA), n_sample2-n_sample1, n_sample2-n_sample1);
            dL = constructL(dD, dA);
        elseif strcmp(what, "bin")
            dD = sparse([1:n_sample2-n_sample1], [1:n_sample2-n_sample1], sum(dA), n_sample2-n_sample1, n_sample2-n_sample1);
            dL = constructL(dD, dA);
        elseif strcmp(what, "abs")
            dD = sparse([1:n_sample2-n_sample1], [1:n_sample2-n_sample1], sum(abs(dA)), n_sample2-n_sample1, n_sample2-n_sample1);
            dL = constructL(dD, dA);
        end
        dL = dL + sparse([1:n_sample2-n_sample1], [1:n_sample2-n_sample1], a0*ones(n_sample2-n_sample1,1), n_sample2-n_sample1, n_sample2-n_sample1);
    end

    nonzerocols = findnnz(L2);
    %loss = @(c)sum(sum((L+c*c').^2));

    V0 = randn(n_samples, kwant);
    %fprintf("nonzeros rate: %10.4f \n", nnz(L)/n_samples/n_samples)
    %fprintf("initial loss: %10.4e \n\n", loss(V0))
    
    
    tic;
    for i = 1:repeat
       [eigV, eigW] = eigs(L1, kwant, 'smallestreal', 'Tolerance', tau);
    end
    timeE = toc/repeat;

    %fprintf("--------------------------------------------\n")
    fprintf("results from eigs(L1): \n")
    fprintf("running time %.4f \n", timeE)
    fprintf("computed eigenvalues: %f \n", sort(diag(eigW))-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigV - eigV*eigW, fnorm)/norm(eigV*eigW, fnorm), loss(eigV))
    fprintf("eigV error: %10.4e \n", norm(L1*eigV - eigV*eigW, 'fro')/norm(eigV*eigW, 'fro'))
    fprintf("--------------------------------------------\n")

    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = chdav(L1, kwant, opts);
    end
    time = toc/repeat;


    %fprintf("--------------------------------------------\n")
    fprintf("results from chdav(L1): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L1*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    fprintf("--------------------------------------------\n")

    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = bchdav(L1, kwant, opts_blk);
    end
    time = toc/repeat;


    %fprintf("--------------------------------------------\n")
    fprintf("results from bchdav(L1): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L1*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    fprintf("--------------------------------------------\n")

    [V, D] = eig(full(L1));
    reald = sort(diag(D));
    fprintf("results from eig: \n")
    fprintf("exact eigenvalues: %f \n", reald(1:10) - a0)
    fprintf("--------------------------------------------\n")


    [dV, dD] = eigs(dL, kwant, 'smallestreal', 'Tolerance', tau);
    V0 = [eigenvectors; dV];
    vV = vecnorm(V0);
    V0 = V0./vV;
    opts1 = opts_blk;
    opts1.v0 = V0;

    
    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = bchdav(L2, kwant, opts1);
    end
    time = toc/repeat;


    %fprintf("--------------------------------------------\n")
    fprintf("results from bchdav(L2, good init): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L2*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    fprintf("--------------------------------------------\n")
    
    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = bchdav(L2, kwant, opts_blk);
    end
    time = toc/repeat;


    %fprintf("--------------------------------------------\n")
    fprintf("results from bchdav(L2, rand init): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L2*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    
end

%exit
