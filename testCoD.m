% This code tests the coordinate descent idea for leading eigenvalues
clc;
clear all;
if 1 % CoD
    N = 1000;
    H = randn(N,N); H = (H' + H)/2;
    s = eig(H);
    H = H - (s(1)-1)*eye(N,N);
    [ut,st] = eig(H);
    st = diag(st);
    
    % gradient descent method for minimizing f(c) = ||H-cc^T||^2_F
    k = 4; % the number of eigenvalues
    c = randn(N,k);%+i*randn(N,k);
    %c = qr(c,0);
    f = @(c) sum(sum((H-c*c').^2));
    % gradient of f(c)
    Grad = @(c) 4*(-H*c + c*(c'*c));
    maxiter = 1000*N;
    gamma = 0.0002; % step size
    loss = zeros(1,maxiter);
    K = c'*c;
    errkeep = [];
    c0 = c;
    K0 = K;
    cn = zeros(1,k);
    for cnt = 1:maxiter
        if 0 % not stable, real CoD
            j = mod(cnt-1,N)+1;
            U = -gamma*4*(-H(j,:)*c(:,:)+c(j,:)*K);
            ct = c';
            G = ct(:,j)*U+U'*c(j,:)+U'*U;
            c(j,:) = c(j,:) + U;
            K = K + G;
            c = c*diag(1./sqrt(sum(c.^2))); % need this for large
            %matrices, otherwise NaN, but this will make the performance
            %worse
        end
        if 1 % stable, row-wise update using old c
            j = mod(cnt-1,N)+1;
            U = -gamma*4*(-H(j,:)*c0(:,:)+c0(j,:)*K0);
            ct = c0';
            G = ct(:,j)*U+U'*c0(j,:)+U'*U;
            c(j,:) = c(j,:) + U;
            cn = cn + c(j,:).^2;
            K = K + G;
            if j == N
                %diag(1./sqrt(sum(c.^2)))*K*diag(1./sqrt(sum(c.^2)))
                dc = diag(1./sqrt(cn)); % need this for large matrices
                K = dc*K*dc;
                K0 = K;
                c = c*dc;
                c0 = c;
                cn = zeros(1,k);
            end
        end
        
        
        if mod(cnt,1000*N)==0
            [cc,~]=qr(c0,0);
            [us,ss] = eig(cc'*H*cc);
            
            es = diag(ss);
            ev =cc*us;
            es = (ev'*H*ev);
            errev = norm(full(ev*es-H*ev))/norm(full(H))
            errkeep = [errkeep,errev]
        end
    end
    [cc,~]=qr(c,0);
    [us,ss] = eig(cc'*H*cc);
    es = diag(ss);
    ev =cc*us;
    es = (ev'*H*ev);
    errev = norm(ev*es-H*ev)/norm(H)
    es = sort(diag(es));
    erres = norm(es-st(end-k+1:end))
end


if 0 % GD, BLAS3, all k eigs together
    N = 1000;
    H = randn(N,N); H = (H' + H)/2;
    s = eig(H);
    H = H - (s(1)-1)*eye(N,N);
    [ut,st] = eig(H);
    st = diag(st);
    
    % gradient descent method for minimizing f(c) = ||H-cc^T||^2_F
    k = 4; % the number of eigenvalues
    c = randn(N,k)+i*randn(N,k);
    %c = qr(c,0);
    f = @(c) sum(sum((H-c*c').^2));
    % gradient of f(c)
    G = @(c) 4*(-H*c + c*(c'*c));
    maxiter = 10000;
    gamma = 0.001; % step size
    loss = zeros(1,maxiter);
    errkeep = [];
    for cnt = 1:maxiter
        c = c - gamma*G(c);
        c = c*diag(1./sqrt(sum(c.^2))); % need this for large matrices
        
        if mod(cnt,1000)==0
            [cc,~]=qr(c,0);
            [us,ss] = eig(cc'*H*cc);
            
            es = diag(ss);
            ev =cc*us;
            es = (ev'*H*ev);
            errev = norm(full(ev*es-H*ev))/norm(full(H))
            errkeep = [errkeep,errev]
        end
    end
    [cc,~]=qr(c,0);
    [us,ss] = eig(cc'*H*cc);
    es = diag(ss);
    ev =cc*us;
    es = (ev'*H*ev);
    errev = norm(ev*es-H*ev)/norm(H)
    es = sort(diag(es));
    erres = norm(es-st(end-k+1:end))
end


% This code tests the coordinate descent idea for leading eigenvalues
if 0 % GD BLAS2, one by one for k eigs
    clc;
    
    N = 20;
    H = randn(N,N); H = (H' + H)/2;
    s = eig(H);
    H = H - (s(1)-1)*eye(N,N);
    [ut,st] = eig(H);
    st = diag(st);
    
    % gradient descent method for minimizing f(c) = ||H-cc^T||^2_F
    k = 4; % the number of eigenvalues
    etr = zeros(N,k); esr = zeros(k,1);
    for cntk = 1:k
        c = rand(N,1);
        c = qr(c,0);
        HH = H;
        for j = 1:cntk-1
            HH = HH - etr(:,j)*etr(:,j)'*esr(j);
            %HH = HH - ut(:,end-j+1)*ut(:,end-j+1)'*st(end-j+1);
        end
        f = @(c) sum(sum((HH-c*c').^2));
        % gradient of f(c)
        G = @(c) 4*(-HH*c + c*diag(diag(c'*c)));
        %G = @(c) 4*(-H*c + (c'*c)*c);
        maxiter = 100000;
        gamma = 0.001; % step size
        %loss = zeros(1,maxiter);
        for cnt = 1:maxiter
            %loss(cnt) = f(c);
            c = c - gamma*G(c);
        end
        ev = c/norm(c);
        err = norm(ev*st(end-cntk+1)-HH*ev)
        etr(:,cntk) = ev;
        esr(cntk) = ev'*HH*ev;
    end
    [esr-st(end:-1:end-k+1)]
end

