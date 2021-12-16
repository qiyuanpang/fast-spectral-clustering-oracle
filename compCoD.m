close all;
startup;

sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
%sizes = [500000, 1000000, 5000000]
kwant = 3
repeat = 1;
what = "bin"

% parameters for Chebyshev-Davidson method
m = 9;
tau = 1e-6
itermax = 200
opts = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1);

% parameters for Coordinate-wise Descent
stepsize = 0.1
w = 0
alpha = 0.9
p = 1.8;
th = 0.0;

a0 = 0.1;
as = struct("abs", [], "pos", [], "bin", []);
bs = struct("abs", [], "pos", [], "bin", []);

% for kwant = 3
%as.abs = [0.1347, 0.1040, 0.0008, 0.00002, 0.00001, 0.00001, 0.00001, 0.00001]+a0;
%as.bin = [0.2182, 0.00001, 0.00001, 0.00009, 0.00001, 0.00001, 0.00001, 0.00001]+a0;
%as.pos = [0.0728, 0.00001, 0.00001, 0.00005, 0.00001, 0.00001, 0.00001, 0.00001]+a0;
b = 1;

as.abs = [0.1650, 0.1350, 0.0020, 0.0011, 0.0080, 0.0023, 0.0443, 0.0010] + a0;
as.pos = [0.1100, 0.0110, 0.0010, 0.0010, 0.0024, 0.0023, 0.0115, 0.0010] + a0;
as.bin = [0.3100, 0.0160, 0.0040, 0.0010, 0.0130, 0.0072, 0.0296, 0.0010] + a0;

SIZES = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000];

for n_samples = sizes
    % n_samples = 5000
    fprintf("\n\n")
    fprintf("========================= #samples = %10d ============================\n", n_samples)
    fname = "sparsedata/" + num2str(n_samples) + "/sparse" + num2str(n_samples) + ".mat"; 
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
        as1 = as.pos;
    elseif strcmp(what, "bin")
        %A = binMat(A);
        %D = constructD(A);
        D = sparse([1:n_samples], [1:n_samples], sum(A), n_samples, n_samples);
        L = constructL(D, A);
        as1 = as.bin;
    elseif strcmp(what, "abs")
        %D = constructD_abs(A);
        D = sparse([1:n_samples], [1:n_samples], sum(abs(A)), n_samples, n_samples);
        L = constructL(D, A);
        as1 = as.abs;
    end

    dL = sparse([1:n_samples], [1:n_samples], a0*ones(n_samples,1), n_samples, n_samples);
    L = L + dL;
    L = (L + L')/2;    
    
    nonzerocols = findnnz(L);

    %loss = @(c)sum(sum((L+c*c').^2));

    V0 = randn(n_samples, kwant);
    fprintf("nonzeros rate: %10.4f \n", nnz(L)/n_samples/n_samples)
    %fprintf("initial loss: %10.4e \n\n", loss(V0))
    
    fprintf("\n\n")

    % tic;
    % for i = 1:repeat
    %     [eigenvalues, eigenvectors, nconv, history] = chdav(L, kwant, opts);
    % end
    % time = toc/repeat;


    % fprintf("--------------------------------------------\n")
    % fprintf("results from chdav: \n")
    % fprintf("running time %.4f \n", time)
    % fprintf("#iteration: %10d \n", history(end,1))
    % fprintf("computed eigenvalues: %f \n", eigenvalues)
    % fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    % %fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)))

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


    tic;
    for i = 1:repeat
        [eigenvalues, eigenvectors] = CoordinateDescent_triofm(L, kwant, stepsize, nonzerocols, itermax, V0, w, alpha);
        %[eigenvalues, eigenvectors] = triofm_org(L, kwant, stepsize, itermaxCoD);
    end
    time = toc/repeat;
    
    fprintf("--------------------------------------------\n")
    fprintf("results from CoD: \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d = %5d * %10d \n", itermaxCoD, itermax, n_samples)
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)))
    
    %as
    %find(SIZES == n_samples)
    a = as1(find(SIZES == n_samples));    
    %b = norm(L,1);
    b = 2.5;
    %a = a0 + (b-a0)/10

    tic;
    for i = 1:repeat
        %[eigenvalues, eigenvectors, ~, newa] = CoordinateDescent_triofm_mod(L, kwant, stepsize, nonzerocols, itermaxCoD, V0, w, alpha, m, a, b, a0, p);
        [eigenvalues, eigenvectors, ~, newa] = CoordinateDescent_triofm_mod2(L, kwant, stepsize, nonzerocols, itermaxCoD, V0, w, alpha, m, a, b, a0, p, th);
        %[eigenvalues, eigenvectors] = triofm_org(L, kwant, stepsize, itermaxCoD);
    end
    time = toc/repeat;
    
    fprintf("--------------------------------------------\n")
    fprintf("results from CoD(mod): \n")
    fprintf("running time %.4f \n", time)
    fprintf("#iteration: %10d = %5d * %10d \n", itermaxCoD, itermax, n_samples)
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues)-a0)
    %fprintf("eigV error: %10.4e , loss: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)), loss(eigenvectors))
    fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues))/norm(eigenvectors*diag(eigenvalues)))

    %a = newa;
    %L = chebfilter2(L, eye(n_samples), m, a, b, a0, p);
    tic;
    for i = 1:repeat
        [eigV, eigW] = eigs(L, kwant, 'largestreal', 'Tolerance', tau);
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
