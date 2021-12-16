close all; clear all; clc;

n_samples = 2000
fname = num2str(n_samples) + "/sim" + num2str(n_samples) + ".json";
fid = fopen(fname, 'r');
raw = fread(fid, inf);
str = char(raw');
A = jsondecode(str);
what = 'abs'
mclc = 1

[n,m] = size(A);
D = zeros(n,m);
switch what
    case 'pos'
        for i = 1:n
            for j = 1:m
                if A(i,j) < 0
                    A(i,j) = 0;
                end
            end
        end
        for i = 1:n
            D(i,i) = sum(A(i,:));
        end
    case 'bin'
        for i = 1:n
            for j = 1:m
                if A(i,j) > 0
                    A(i,j) = 1;
                else
                    A(i,j) = 0;
                end
            end
        end
        for i = 1:n
            D(i,i) = sum(A(i,:));
        end
    case 'abs'
        for i = 1:n
            D(i,i) = sum(abs(A(i,:)));
        end
    case 'org'
        for i = 1:n
            D(i,i) = sum(A(i,:));
        end
end

Dinv = diag(1./diag(D));
D2inv = diag(1./sqrt(diag(D)));
L = eye(n) - D2inv*A*D2inv;
%L = D - A;
P = Dinv*A;

switch mclc
    case 1
        L = L + 0.1*eye(n);
        tic;
        for i = 1:5
           [vec, val] = eigs(L,2,'smallestabs');
           vec = vec(:,2);
           vec = vec/norm(vec);
           val = diag(val);
           val = val(2);
           idx = sign(vec-median(vec));
        end
        time = round(toc/5,2);
    case 2
        tic;
        for i = 1:5
           [vec, val] = eigs(P,1,'largestabs');
           vec = real(vec);
           vec = vec/norm(vec);
           idx = sign(vec - median(vec));
        end
        time = round(toc/5,2);
end

val, min(vec), max(vec)
idx(idx==-1) = 0;
savefile = "labels_" + num2str(n_samples) + "_2_2_" + what + "_" + num2str(mclc) + ".json";
fid = fopen(savefile, 'w');
labels = struct("labels", idx, "time", time, "vec", real(vec), "val", real(val));
encodedlabels = jsonencode(labels);
fprintf(fid, encodedlabels);
exit
