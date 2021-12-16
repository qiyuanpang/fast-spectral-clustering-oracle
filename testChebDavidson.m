close all; clear all; clc;

n_samples = 2000
fname = num2str(n_samples) + "/sim" + num2str(n_samples) + ".json";
fid = fopen(fname, 'r');
raw = fread(fid, inf);
str = char(raw');
A = jsondecode(str);

x = randn(n_samples, 1);
m = 5;
kwant = 2;
kkeep = 1;
dimmax = 3;
tau = 1e-3;
itermax = 20;

tic;
for i = 1:5
    [eigvalues, eigvectors] = ChebDavidson(A, x, m, kwant, kkeep, dimmax, tau, itermax);
end
time = toc/5;

print(time)
print(eigenvalues)
print(eigenvectors)
