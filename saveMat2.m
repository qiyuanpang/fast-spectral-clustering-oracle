close all;
startup;

sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
%sizes = [5000000]
kwant = 4
repeat = 1;
what = "abs"

% parameters for Chebyshev-Davidson method
m = 9;
tau = 1e-6
itermax = 200
opts = struct('polym', m, 'tol', tau, 'itmax', itermax, 'chksym', 1, 'nomore', 1);

% parameters for Coordinate-wise Descent
stepsize = 0.1
w = 0
alpha = 0.9

%ncores = 34
%parpool(ncores, 'IdleTimeout', 360);
for n_samples = sizes
    % n_samples = 5000
    fprintf("\n\n")
    fprintf("========================= #samples = %10d ============================\n", n_samples)
    fname = "sparsedata/" + num2str(n_samples) + "/sparse" + num2str(n_samples) + what +".mat";      
    
    A = load(fname);
    A = A.A;
   
    %nnz(abs(A) > 1e-10)/n_samples/n_samples 
    save(fname, 'A', '-v7.3');
    fprintf("%s modified and saved! \n", fname);
end

exit
