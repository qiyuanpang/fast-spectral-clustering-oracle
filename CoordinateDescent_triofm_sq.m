function [eigvalues, eigvectors, iter] = CoordinateDescent_triofm_sq(A, k, gamma, nonzerocols, itermax, V, w, alpha, tol, batch)
    [N,~] = size(A);
    K = V'*V;
    K0 = K;
    V0 = V;
    iter = 1;
    dU = zeros(N,k);
    AV0 = A*V0;
    conv = 0;
    for iter = 1:itermax
        U = -gamma*4*(AV0 + V0*((1-w)*triu(K0) + w*K0));
        %G = V0'*U + U'*V0 + U'*U;
        V = V + U + alpha*dU;
        %V = V + U;
        K = V'*V;
        r = diag(K);
        Dinv = diag(1./sqrt(r));
        K = Dinv*K*Dinv;
        V = V*Dinv;
        K0 = K;
        V0 = V;
        AV0 = A*V0;
        dU = U;
        if mod(iter, batch) == 0
            [V1,~] = qr(V,0);
            [Q, D] = eig(V1'*A*V1);
            if norm(A*V1*Q - V1*Q*D, 'fro') <= tol*norm(A*V1*Q, 'fro')
                eigvectors = V1*Q;
                eigvalues = diag(D);
                conv = 1;
                break
            end
        end
    end
    if conv == 0
        [V,~] = qr(V,0);
        H = V'*A*V;
        [Q, D] = eig(H);
        eigvectors = V*Q;
        eigvalues = diag(D);
    end
end
