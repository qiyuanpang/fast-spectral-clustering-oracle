function [eigvalues, eigvectors, iter, ck] = CoordinateDescent_triofm_sq_orth(A, k, gamma, nonzerocols, itermax, V, w, alpha, tol, batch)
    [N,~] = size(A);
    K = V'*V;
    K0 = K;
    V0 = V;
    iter = 1;
    dU = zeros(N,k);
    AV0 = A*V0;
    ck = 0;
    conv = 0;
    eigvalues = zeros(k, 1);
    for iter = 1:itermax
        wK = (1-w)*triu(K0) + w*K0;
        U = -gamma*4*(AV0(:,ck+1:end) + V0(:,ck+1:end)*wK(ck+1:end,ck+1:end));
        %U = -gamma*4*(AV0(:,ck+1:end) + V0*wK(:,ck+1:end));
        V(:,ck+1:end) = V(:,ck+1:end) + U + alpha*dU(:,ck+1:end);
        %V(:,ck+1:end) = V(:,ck+1:end) + U;
        K(ck+1:end,ck+1:end) = V(:,ck+1:end)'*V(:,ck+1:end);
        %K = V'*V;
        r = diag(K(ck+1:end,ck+1:end));
        Dinv = diag(1./sqrt(r));
        K(ck+1:end,ck+1:end) = Dinv*K(ck+1:end,ck+1:end)*Dinv;
        V(:,ck+1:end) = V(:,ck+1:end)*Dinv;
        dU(:,ck+1:end) = U;
        if mod(iter, batch) == 0
            [Q,R] = qr(V(:,ck+1:end), 0);
            Y1 = A*Q;
            [W,D] = eig(Q'*Y1);
            V1 = Q*W;
            Y1 = Y1*W;
            d = find(vecnorm(Y1-V1*D) <= tol*vecnorm(V1*D));
            dd = diag(D);
            if length(d) > 0
                cols = [1:k-ck];
                cplm = setdiff(cols, d);
                V(:,ck+[1:length(d)]) = V1(:,d);
                eigvalues(ck+[1:length(d)]) = dd(d);
                ckold = ck;
                ck = ck + length(d);
                if ck == k
                    eigvectors = V;
                    conv = 1;
                    break
                end
                V(:,ck+1:end) = V1(:,cplm);
                Knew = W'*(R'\(K(ckold+1:end,ckold+1:end)/R))*W;
                K(ck+1:end,ck+1:end) = Knew(cplm,cplm);
                
                %K(ck+1:end,ck+1:end) = V(:,ck+1:end)'*V(:,ck+1:end);
                %K = V'*V;
                dU(:,ck+1:end) = zeros(N,k-ck);
            end
        end
        K0(ck+1:end,ck+1:end) = K(ck+1:end,ck+1:end);
        V0(:,ck+1:end) = V(:,ck+1:end);
        AV0(:,ck+1:end) = A*V0(:,ck+1:end);
    end
    if conv == 0
        [V,~] = qr(V,0);
        H = V'*A*V;
        [Q, D] = eig(H);
        eigvectors = V*Q;
        eigvalues = diag(D);
    end
end
