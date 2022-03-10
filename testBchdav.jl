using LinearAlgebra
using Printf
include("./Bchdav/bchdav.jl")
using .Bchdav
using MAT
using Arpack
using CPUTime
using SparseArrays
include("./utils/constructL.jl")

sizes = [1000, 10000, 100000, 1000000, 10000000, 50000000]
kwant = 5
repeat = 1

what = "abs"

m = 11
tau = 1e-3
itermax = 500
a0 = 0.1

opts = Dict([("polym", m), ("tol", tau), ("itmax", itermax), ("chksym", true), ("kmore", 0), ("blk", kwant), ("upb", 2.1+a0)])

batch = 50
fnorm = "fro"

@printf("type: %s \n", what)

for i = 1:length(sizes)
    n_samples = sizes[i]
    @printf("\n\n")
    @printf("========================= #samples = %10d ============================\n", n_samples)

    fname = "sparsedata/sparse" * string(n_samples) * "/sparse" * string(n_samples) * what * ".mat"
    file = matopen(fname)
    A = read(file, "A")
    A = (A+A')/2

    if  what == "pos"
        D = sparse(collect(1:n_samples), collect(1:n_samples), dropdims(sum(A,dims=1), dims=1), n_samples, n_samples);
        L = constructL(D, A);
    elseif what == "bin"
        D = sparse(collect(1:n_samples), collect(1:n_samples), dropdims(sum(A, dims=1),dims=1), n_samples, n_samples);
        L = constructL(D, A);
    elseif what == "abs"
        D = sparse(collect(1:n_samples), collect(1:n_samples), dropdims(sum(broadcast(abs,A),dims=1), dims=1), n_samples, n_samples);
        L = constructL(D, A);
    end

    N = n_samples
    dL = sparse(collect(1:N), collect(1:N), a0*ones(N), N, N)
    L = L + dL
    L = (L+L')/2
    
    
    CPUtic()
    #for j = 1:repeat
    evals, eigV, kconv, history = bchdav(L, kwant, opts)
    #end
    cputime = CPUtoq()/repeat

    @printf("results from bchdav: \n")
    @printf("running time %.4f \n", cputime)
    @printf("iteration: %10d \n", history[end,1])
    for j = 1:length(evals)
        @printf("%3i-th eigenvalue computed: %f \n", i, evals[j]-a0)
    end
    @printf("eigV error: %10.4e \n", norm(L*eigV - eigV*Diagonal(evals), 2)/norm(eigV*Diagonal(evals), 2))
    @printf("--------------------------------------------\n")

    
    CPUtic()
    #for j = 1:repeat
    evals, eigV = eigs(L, nev=kwant, which=:SR, tol=tau)
    #end
    cputime = CPUtoq()/repeat

    @printf("results from eigs(Arpack): \n")
    @printf("running time %.4f \n", cputime)
    #@printf("iteration: %10d \n", history[end,1])
    for j = 1:length(evals)
        @printf("%3i-th eigenvalue computed: %f \n", i, evals[j]-a0)
    end
    @printf("eigV error: %10.4e \n", norm(L*eigV - eigV*Diagonal(evals), 2)/norm(eigV*Diagonal(evals), 2))


    flush(stdout)     
end

