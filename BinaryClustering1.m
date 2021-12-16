close all; clear all; clc;

% TODO 0: try cross-clusters; only have a few columns of A, global sparse case,
% 10m rows and some columns, try to speed up; local sparse case, 10m rows,
% but the column indices for each row we have are different, this example
% comes from the kNN sparse similarity, we have value and index talk and
% skiny matrices. in computation, the k nn might be infered from different
% columns of information. fort example, 2 & 3 are close and indicated for
% point 2 but not for point 3. then when compute with point 3, we can use
% the information from pint 2.

% choose different eigensolvers, msol
% 1. eigs in matlab
% 2. Chebyshev preconditioner
% 3. coordinate descent with adaptive entry-wise update, TODO 3

% choose different clustering methods, mclc
% mclc = 1: symmetric normalized Laplaican
% mclc = 2: random walk normalized adjacency matrix

mclc = 2

% TODO 4: bi-clustering for a few times v.s. clustering with k eigenpairs
% TODO 5: whether we use k-means in the second step;

% prepare a toy example
fig = figure()
subplot(2,2,1);
npts = 100;
x1 = 1 + randn(npts,1)/9;
x2 =  randn(npts,1)/9;
x = [x1,x2];
plot(x1,x2,'or'); hold on;
x1 =  2 + randn(npts,1)/9;
x2 =  randn(npts,1)/9;
y = [x1,x2];
x = [x;y];
plot(x1,x2,'*b');
x1 =  1 + randn(npts,1)/9;
x2 =  1 + randn(npts,1)/9;
y = [x1,x2];
x = [x;y];
plot(x1,x2,'+g');
x1 =  randn(npts,1)/9;
x2 =  randn(npts,1)/9;
y = [x1,x2];
x = [x;y];
plot(x1,x2,'xy');
x1 =  randn(npts,1)/9;
x2 =  1+randn(npts,1)/9;
y = [x1,x2];
x = [x;y];
plot(x1,x2,'sc');
x1 =  randn(npts,1)/9;
x2 =  2 + randn(npts,1)/9;
y = [x1,x2];
x = [x;y];
plot(x1,x2,'>m');
x1 =  1+randn(npts,1)/9;
x2 =  2+randn(npts,1)/9;
y = [x1,x2];
x = [x;y];
plot(x1,x2,'<k');
x1 =  2+randn(npts,1)/9;
x2 =  2+randn(npts,1)/9;
y = [x1,x2];
x = [x;y];
plot(x1,x2,'db');

axis square;
title('Given clusters')

% compute the similarity matrix A and the matrix L
% TODO 1: design a matrix-free method so that we don't need to generate A,
% Lnorm, and P
npts=size(x,1);
sigma = 0.1;
A=zeros(npts,npts);
for i=1:npts
    for j=1:npts
        A(i,j) = exp(-norm(x(i,:)-x(j,:))^2/sigma);
    end
end
% binary example
% A = zeros(npts,npts);
% A(1:npts/4,1:npts/4) = 1;
% A(npts/2+1:end,npts/2+1:end) = 1;
% A(npts/4+1:npts/2,npts/4+1:npts/2) = 1;


D2inv = diag(1./sqrt(sum(A,2)));
Dinv = diag(1./(sum(A,2)));
Lnorm = eye(npts) - D2inv*A*D2inv; % symmetric normalized Laplacian
P = Dinv*A; % random walk normalized adjacency matrix


% find eigen pairs
% TODO 2: Lnorm matrix is too singular and needs preconditioning
switch mclc
    case 1
        [vec,val] = eigs(Lnorm,2,'smallestabs');
        vec = vec(:,2);
        idx = sign(vec - median(vec));
    case 2
        tic;
        for i = 1:10
            [vecr,valr] = eigs(P,3); % working
            idx1 = sign(vecr(:,1) - median(vecr(:,1)));
            m1 = median(vecr(find(idx1==1),2));
            m2 = median(vecr(find(idx1==-1),2));
            idx2 = zeros(npts);
            for j = 1:npts
                if idx1(j) == 1
                   idx2(j) = sign(vecr(j,2)-m1);
                else
                   idx2(j) = sign(vecr(j,2)-m2);
                end
            end
            m11 = median(vecr(find(idx1==1 & idx2==1),3));
            m12 = median(vecr(find(idx1==1 & idx2==-1),3));
            m21 = median(vecr(find(idx1==-1 & idx2==1),3));
            m22 = median(vecr(find(idx1==-1 & idx2==-1),3));
            idx3 = zeros(npts);
            for j = 1:npts
                if idx1(j)==1 & idx2(j)==1
                   idx3(j) = sign(vecr(j,3)-m11);
                elseif idx1(j)==1 & idx2(j)==-1
                   idx3(j) = sign(vecr(j,3)-m12);
                elseif idx1(j)==-1 & idx2(j)==1
                   idx3(j) = sign(vecr(j,3)-m21);
                else
                   idx3(j) = sign(vecr(j,3)-m22);
                end
            end
        end
        time = toc/10;
end

% visualize binary clustering
subplot(2,2,2);
loc = find(idx1==1 & idx2==1 & idx3==1);
plot(x(loc,1),x(loc,2),'or'); hold on;
loc = find(idx1==1 & idx2==1 & idx3==-1);
plot(x(loc,1),x(loc,2),'ob'); hold on;
loc = find(idx1==1 & idx2==-1 & idx3==1);
plot(x(loc,1),x(loc,2),'*y'); hold on;
loc = find(idx1==1 & idx2==-1 & idx3==-1);
plot(x(loc,1),x(loc,2),'+m'); hold on;
loc = find(idx1==-1 & idx2==1 & idx3==1);
plot(x(loc,1),x(loc,2),'sc'); hold on;
loc = find(idx1==-1 & idx2==1 & idx3==-1);
plot(x(loc,1),x(loc,2),'dk'); hold on;
loc = find(idx1==-1 & idx2==-1 & idx3==1);
plot(x(loc,1),x(loc,2),'>g'); hold on;
loc = find(idx1==-1 & idx2==-1 & idx3==-1);
plot(x(loc,1),x(loc,2),'^b'); hold on;
axis square;
title("Identified clusters (Binary," + num2str(time) + ")")

tic;
for i = 1:10
    [vecr,valr] = eigs(P,3);
    idx = kmeans(vecr, 8);
end
time = toc/10;
subplot(2,2,3);
loc = find(idx==1);
plot(x(loc,1),x(loc,2),'or'); hold on;
loc = find(idx==2);
plot(x(loc,1),x(loc,2),'^g'); hold on;
loc = find(idx==3);
plot(x(loc,1),x(loc,2),'>b'); hold on;
loc = find(idx==4);
plot(x(loc,1),x(loc,2),'<y'); hold on;
loc = find(idx==5);
plot(x(loc,1),x(loc,2),'sc'); hold on;
loc = find(idx==6);
plot(x(loc,1),x(loc,2),'dk'); hold on;
loc = find(idx==7);
plot(x(loc,1),x(loc,2),'+m'); hold on;
loc = find(idx==8);
plot(x(loc,1),x(loc,2),'+r'); hold on;
axis square;
title("Identified clusters (8-means," + num2str(time) + ")")
saveas(fig, 'sample.png')
exit
