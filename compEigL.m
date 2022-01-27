close all;
startup;

sizes = [500000, 1000000, 5000000]
%sizes = [5000000]
kwant = 50
repeat = 1;
what = "abs"

% parameters for Chebyshev-Davidson method
m = 15;
tau = 1e-6
itermax = 9000
opts = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1, 'vmax', 900);

for n_samples = sizes
    
    fprintf("\n\n")
    fprintf("========================= #samples = %10d ============================\n", n_samples)
    
    fname = "sparsedata/" + num2str(n_samples) + "/sparse" + num2str(n_samples) + what + ".mat";
    data = load(fname);
    A = data.A;
    A = (A+A.')/2;

    if strcmp(what, "pos")
        D = sparse([1:n_samples], [1:n_samples], sum(A), n_samples, n_samples);
        %L = constructL(D, A);
    elseif strcmp(what, "bin")
        D = sparse([1:n_samples], [1:n_samples], sum(A), n_samples, n_samples);
        %L = constructL(D, A);
    elseif strcmp(what, "abs")
        D = sparse([1:n_samples], [1:n_samples], sum(abs(A)), n_samples, n_samples);
        %L = constructL(D, A);
    end

    L = D - A;
    % L = constructL(D, A);


    [eigenvalues, eigenvectors, nconv, history] = chdav(L, kwant, opts);
    fprintf("computed eigenvalues: %f \n", sort(eigenvalues))
    fprintf("eigV error: %10.4e \n", norm(L*eigenvectors - eigenvectors*diag(eigenvalues), 'fro')/norm(eigenvectors*diag(eigenvalues), 'fro'))
end
