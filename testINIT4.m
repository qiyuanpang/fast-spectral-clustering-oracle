close all;
startup;

sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
%sizes = [5000000]
kwant = 5
repeat = 1;
what = "abs"

% parameters for Chebyshev-Davidson method
m = 5;
tau = 1e-3
itermax = 3000


% parameters for Coordinate-wise Descent
stepsize = 0.02
w = 1.0
alpha = 0.9
th = 0.3;
a0 = 0.1;
p = 1.5;

opts = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1);
opts_blk = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'kmore', 0, 'blk', kwant, 'upb', 3+a0);

ncores = 40
parpool(ncores, 'IdleTimeout', 360);

batch = 50;
fnorm = 'fro';
for n_samples = sizes
    % n_samples = 5000
    fprintf("\n\n")
    fprintf("========================= #samples = %10d ============================\n", n_samples)
    fname = "sparsedata/" + num2str(n_samples) + "/sparse" + num2str(n_samples) + what + ".mat"; 

    A = load(fname);
    A = A.A;
    
    A = (A+A')/2;

    itermaxCoD = itermax*n_samples;

    if strcmp(what, "pos")
        D = sparse([1:n_samples], [1:n_samples], sum(A), n_samples, n_samples);
        L = constructL(D, A);
    elseif strcmp(what, "bin")
        D = sparse([1:n_samples], [1:n_samples], sum(A), n_samples, n_samples);
        L = constructL(D, A);
    elseif strcmp(what, "abs")
        D = sparse([1:n_samples], [1:n_samples], sum(abs(A)), n_samples, n_samples);
        L = constructL(D, A);
    end

    dL = sparse([1:n_samples], [1:n_samples], a0*ones(n_samples,1), n_samples, n_samples);
    L = L + dL;
    L = (L + L')/2;
    
    nonzerocols = findnnz(L);
    fprintf("nonzeros rate: %10.4f \n", nnz(L)/n_samples/n_samples)
    %loss = @(c)sum(sum((L+c*c').^2));


    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = bchdav(L, kwant, opts_blk);
    end
    time = toc/repeat;


    %fprintf("--------------------------------------------\n")
    fprintf("results from bchdav(L): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    fprintf("--------------------------------------------\n")



    idx = randi([1, n_samples], floor(n_samples*0.05), 2);
    val = rand(floor(n_samples*0.05), 1);

    for i = 1:size(idx,1)
        row = idx(i, 1);
        col = idx(i, 2);
        vl = val(i);
        if strcmp(what, "bin")
            vl = 1;
        end
        if abs(row - col) > 0
            A(row, col) = vl;
            A(col, row) = vl;
        end
    end


    if strcmp(what, "pos")
        D1 = sparse([1:n_samples], [1:n_samples], sum(A), n_samples, n_samples);
        L1 = constructL(D1, A);
    elseif strcmp(what, "bin")
        D1 = sparse([1:n_samples], [1:n_samples], sum(A), n_samples, n_samples);
        L1 = constructL(D1, A);
    elseif strcmp(what, "abs")
        D1 = sparse([1:n_samples], [1:n_samples], sum(abs(A)), n_samples, n_samples);
        L1 = constructL(D1, A);
    end

    dL = sparse([1:n_samples], [1:n_samples], a0*ones(n_samples,1), n_samples, n_samples);
    L1 = L1 + dL;
    L1 = (L1 + L1')/2;


    V0 = eigenvectors;
    opts1 = opts_blk;
    opts1.v0 = V0;
    
    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, nconv, history] = bchdav(L1, kwant, opts1);
    end
    time = toc/repeat;


    %fprintf("--------------------------------------------\n")
    fprintf("results from bchdav(L1, good init): \n")
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
    fprintf("results from bchdav(L1, rand init): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L1*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    fprintf("--------------------------------------------\n")


    
    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, iters] = CoordinateDescent_triofm_par(L1, kwant, stepsize, nonzerocols, itermax, V0, w, alpha, tau, batch);
    end
    time = toc/repeat;

    %fprintf("--------------------------------------------\n")
    fprintf("results from CoD(par, good init): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d = %5d * %10d \n", iters*n_samples, iters, n_samples)
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues), fnorm)/norm(eigenvectors*diag(eigenvalues), fnorm), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L1*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    fprintf("--------------------------------------------\n")
    
    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors, iters] = CoordinateDescent_triofm_par(L1, kwant, stepsize, nonzerocols, itermax, rand(n_samples, kwant), w, alpha, tau, batch);
    end
    time = toc/repeat;

    %fprintf("--------------------------------------------\n")
    fprintf("results from CoD(par, rand init): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d = %5d * %10d \n", iters*n_samples, iters, n_samples)
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues), fnorm)/norm(eigenvectors*diag(eigenvalues), fnorm), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L1*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))


end

exit
