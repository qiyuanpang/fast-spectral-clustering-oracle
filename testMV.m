clear all; close all;

what = "abs"
SIZE = 1000000

fname = "sparsedata/" + num2str(SIZE) + "/sparse" + num2str(SIZE) + what + ".mat"; 
A = load(fname);
A = A.A;
    
A = (A+A')/2;


v = rand(SIZE,10);

%parpool(40);

tic;
y1 = A*v;
toc

tic;
y2 = par_Mv(A,v,40);
toc

norm(y1-y2)/norm(y2)
