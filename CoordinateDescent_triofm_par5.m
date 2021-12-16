function [eigvalues, eigvectors, iter] = CoordinateDescent_triofm_par5(A, k, gamma, nonzerocols, itermax, V, w, alpha, m, a, b, a0, p, tol)
    [N,~] = size(A);
    K = V'*V;
    r = zeros(1,k);
    K0 = K;
    V0 = V;
    iter = 1;
    dU = zeros(N, k);
    AV0 = chebfilter2(A, V0, m, a, b, a0, p);
    dr = zeros(N,k);
    ck = 0;
    for iter = 1:itermax
        G = zeros(k-ck,k-ck,N);
        parfor j = 1:N
            % j = mod(iter-1, N) + 1;
            %cj = nonzerocols{j};
            %U = -gamma*4*(A(j, cj)*V0(cj,:) + V0(j,:)*((1-w)*triu(K0) + w*K0));
            Kw = (1-w)*triu(K0) + w*K0;
            U = -gamma*4*(AV0(j,ck+1:end) + V0(j,ck+1:end)*Kw(ck+1:end,ck+1:end);
            G(:,:,j) = (V0(j,ck+1:end))'*U + U'*V0(j,ck+1:end) + U'*U;
            V(j,ck+1:end) = V(j,ck+1:end) + U + alpha*dU(j,ck+1:end);
            dr(j,ck+1:end) = V(j,ck+1:end).^2;
            dU(j,ck+1:end) = U;
        end
        K(ck+1:end,ck+1:end) = K(ck+1:end:ck+1:end) + sum(G,3);
        r(ck+1:end) = r(ck+1:end) + sum(dr(:,ck+1:end),1);
        Dinv = diag(1./sqrt(r(ck+1:end)));
        K(ck+1:end,ck+1:end) = Dinv*K(ck+1:end,ck+1:end)*Dinv;
        K0(ck+1:end,ck+1:end) = K(ck+1:end,ck+1:end);
        V(:,ck+1:end) = V(:,ck+1:end)*Dinv;
        Y = A*V(:,ck+1:end);
        [~, D] = eig(V(:,ck+1:end)'*Y);
        d = diag(D);
        [eigmin, idx] = min(d);
        if norm(Y(:,idx) - eigmin*V(:,idx)) <= tol*norm(eigmin*V(:,idx))
            tmp = V(:,ck+1);
            V(:,ck+1) = V(:,idx);
            V(:,idx) = tmp;
            ck = ck + 1;
        end
        if ck == k
            break
        end
        a1 = max(d)+min(abs(diff(d)));
        if a1 >= a0 && a1 <= b
            a = a1;
        end
        V0(:,ck+1:end) = V(:,ck+1:end);
        %AV0 = A*V0;
        AV0 = chebfilter2(A, V0, m, a, b, a0, p);
        r = zeros(1,k);
    end
    [V,~] = qr(V,0);
    H = V'*A*V;
    [Q, D] = eig(H);
    eigvectors = V*Q;
    eigvalues = diag(D);
end
