close all;
startup;

sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
%sizes = [5000000]
kwant = 2
repeat = 1;
what = "bin"

% parameters for Chebyshev-Davidson method
m = 5;
tau = 1e-3
itermax = 2000


% parameters for Coordinate-wise Descent
stepsize = 0.02
w = 1.0
alpha = 0.9
th = 0.3;
a0 = 0.1;
p = 1.5;

% opts = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1);
% opts_blk = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'kmore', 0, 'blk', kwant, 'upb', 3+a0);

ncores = 40
parpool(ncores, 'IdleTimeout', 360);

batch = 25;
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
        [eigenvalues, eigenvectors, cl, iter, iter_tl] = CoordinateDescent_sc_par(L, kwant, stepsize, nonzerocols, itermax, rand(n_samples,kwant), w, alpha, tau, batch);
    end
    time = toc/repeat;
    
    cl(find(cl==-1)) = 2;

    cl1 = kmeans(eigenvectors(:,kwant), 2);

    %fprintf("--------------------------------------------\n")
    fprintf("results from CoD_sc: \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d / %10d / %10d \n", iter_tl, iter, floor(iter_tl/iter))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    fprintf("--------------------------------------------\n")




    tic;
    for i = 1:repeat
        [eigenvectors, D] = eigs(L, kwant, 'smallestreal', 'Tolerance', 1e-6, 'MaxIterations', 1000);
    end
    time = toc/repeat;
    eigenvalues = diag(D);


    %fprintf("--------------------------------------------\n")
    fprintf("results from bchdav(L, rand init): \n")
    fprintf("running time %.4f \n", time)
    % fprintf("#iteration: %10d \n", history(end,1))
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
    % fprintf("--------------------------------------------\n")

    cl2 = kmeans(eigenvectors(:,kwant), 2);


    a1 = nnz(find(cl == cl2))/n_samples;
    a2 = nnz(find(cl ~= cl2))/n_samples;
    fprintf("cl  v.s. cl2: %3.2f \n", max(a1,a2))

    a1 = nnz(find(cl1 == cl2))/n_samples;
    a2 = nnz(find(cl1 ~= cl2))/n_samples;
    fprintf("cl1 v.s. cl2: %3.2f \n", max(a1,a2))

     

end

exit
