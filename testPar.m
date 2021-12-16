close all;
startup;

sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
%sizes = [5000000]
kwant = 3
repeat = 1;
what = "abs"

% parameters for Chebyshev-Davidson method
m = 5;
tau = 1e-6
itermax = 200
opts = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1);

% parameters for Coordinate-wise Descent
stepsize = 0.01
w = 0
alpha = 0.9
th = 0.3;
a0 = 0.1;
p = 1.8;

ncores = 40
parpool(ncores, 'IdleTimeout', 360);
for n_samples = sizes
    % n_samples = 5000
    fprintf("\n\n")
    fprintf("========================= #samples = %10d ============================\n", n_samples)
    fname = "sparsedata/" + num2str(n_samples) + "/sparse" + num2str(n_samples) + what + ".mat"; 
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

    % tic;
    % for i = 1:repeat
    %     [eigenvalues, eigenvectors, nconv, history] = chdav(L, kwant, opts);
    % end
    % time = toc/repeat;

    % fprintf("\n\n")

    % fprintf("--------------------------------------------\n")
    % fprintf("results from chdav: \n")
    % fprintf("running time %.4f \n", time)
    % fprintf("#iteration: %10d \n", history(end,1))
    % fprintf("computed eigenvalues: %f \n", eigenvalues-a0)
    % fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    % %fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)))
    
    tic;
    for i = 1:repeat
        %[eigenvalues, eigenvectors] = CoordinateDescent_triofm_par3(L, kwant, stepsize, nonzerocols, itermax, V0, w, alpha,th);
        [eigenvalues, eigenvectors] = CoordinateDescent_triofm_par(L, kwant, stepsize, nonzerocols, itermax, V0, w, alpha);
        %[eigenvalues, eigenvectors] = triofm_org(L, kwant, stepsize, itermaxCoD);
    end
    time = toc/repeat;

    fprintf("--------------------------------------------\n")
    fprintf("results from CoD(par): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d = %5d * %10d \n", itermaxCoD, itermax, n_samples)
    fprintf("computed eigenvalues: %f \n", eigenvalues-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)))


    b = norm(L,1);
    a = a0 + (b-a0)/20;
    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors] = CoordinateDescent_triofm_par4(L, kwant, stepsize, nonzerocols, itermax, V0, w, alpha, m, a, b, a0, p);
        %[eigenvalues, eigenvectors] = CoordinateDescent_triofm_par(L, kwant, stepsize, nonzerocols, itermax, V0, w, alpha);
        %[eigenvalues, eigenvectors] = triofm_org(L, kwant, stepsize, itermaxCoD);
    end
    time = toc/repeat;

    fprintf("--------------------------------------------\n")
    fprintf("results from CoD(par, filter): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d = %5d * %10d \n", itermaxCoD, itermax, n_samples)
    fprintf("computed eigenvalues: %f \n", eigenvalues-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)))



    tic;
    for i = 1:repeat
        [eigV, eigW] = eigs(L, kwant, 'smallestabs', 'Tolerance', tau);
    end
    timeE = toc/repeat;

    fprintf("--------------------------------------------\n")
    fprintf("results from eigs: \n")
    fprintf("running time %.4f \n", timeE)
    fprintf("computed eigenvalues: %f \n", diag(eigW)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigV - eigV*eigW)/norm(eigV*eigW), loss(eigV))
    fprintf("eigV error: %10.4e \n", norm(L*eigV - eigV*eigW)/norm(eigV*eigW))
end

exit
