close all;
startup;

SIZE = 5000000;
sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000, 3000000, 4000000]
%sizes = [5000000]
kwant = 4
repeat = 1;
what = "abs"

% parameters for Chebyshev-Davidson method
m = 9;
tau = 1e-6
itermax = 3000
opts = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1, 'vmax', 50);

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

    %dL = L2(n_sample1+1:end, n_sample1+1:end);
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
    
    
    %tic;
    %for i = 1:repeat
    %    [eigV, eigW] = eigs(L1, kwant, 'smallestreal', 'Tolerance', 1e-10);
    %end
    %timeE = toc/repeat;

    %fprintf("--------------------------------------------\n")
    %fprintf("results from eigs: \n")
    %fprintf("running time %.4f \n", timeE)
    %fprintf("computed eigenvalues: %f \n", sort(diag(eigW))-a0)
    %%fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigV - eigV*eigW, fnorm)/norm(eigV*eigW, fnorm), loss(eigV))
    %fprintf("eigV error: %10.4e \n", norm(L1*eigV - eigV*eigW, 'fro')/norm(eigV*eigW, 'fro'))


    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = chdav(L1, kwant, opts);
    end
    time = toc/repeat;


    fprintf("--------------------------------------------\n")
    fprintf("results from chdav: \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L1*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))

    [dV, dD] = eigs(dL, kwant, 'smallestreal', 'Tolerance', 1e-10);

    V0 = [eigenvectors; dV];
    vV = vecnorm(V0);
    V0 = V0./vV;
    opts1 = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1, 'V0', V0);
    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = chdav(L2, kwant, opts1);
    end
    time = toc/repeat;


    fprintf("--------------------------------------------\n")
    fprintf("results from chdav(init): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L2*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))

    
    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = chdav(L2, kwant, opts);
    end
    time = toc/repeat;


    fprintf("--------------------------------------------\n")
    fprintf("results from chdav: \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L2*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))

end

exit
