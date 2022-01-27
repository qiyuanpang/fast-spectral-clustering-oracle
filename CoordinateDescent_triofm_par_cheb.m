function [eigvalues, eigvectors, iter] = CoordinateDescent_triofm_par_cheb(A, k, gamma, nonzerocols, itermax, V, w, alpha, m, a, b, a0, p, tol, batch)
    [N,~] = size(A);
    K = V'*V;
    r = zeros(1,k);
    K0 = K;
    V0 = V;
    iter = 1;
    dU = zeros(N,k);
    AV0 = chebfilter2(A, V0, m, a, b, a0, p);
    G = zeros(k,k,N);
    dr = zeros(N,k);
    conv = 0;
    for iter = 1:itermax
        parfor j = 1:N
            U = -gamma*4*(AV0(j,:) + V0(j,:)*((1-w)*triu(K0) + w*K0));
            G(:,:,j) = (V0(j,:))'*U + U'*V0(j,:) + U'*U;
            V(j,:) = V(j,:) + U + alpha*dU(j,:);
            %V(j,:) = V(j,:) + U;
            dr(j,:) = V(j,:).^2;
            dU(j,:) = U;
        end
        K = K + sum(G,3);
        r = r + sum(dr,1);
        Dinv = diag(1./sqrt(r));
        K = Dinv*K*Dinv;
        K0 = K;
        V = V*Dinv;
        V0 = V;
        r = zeros(1,k);
        if mod(iter, batch) == 0
            [V1,~] = qr(V,0);
            [Q, D] = eig(V1'*A*V1);
            V1 = V1*Q;
            if norm(A*V1 - V1*D, 'fro') <= tol*norm(A*V1, 'fro')
                eigvectors = V1;
                eigvalues = diag(D);
                conv = 1;
                break
            end
        end
        AV0 = chebfilter2(A, V0, m, a, b, a0, p);
    end
    if conv == 0
        [V,~] = qr(V,0);
        H = V'*A*V;
        [Q, D] = eig(H);
        eigvectors = V*Q;
        eigvalues = diag(D);
    end
end
