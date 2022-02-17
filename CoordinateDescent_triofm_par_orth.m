function [eigvalues, eigvectors, iter, ck] = CoordinateDescent_triofm_par_orth(A, k, gamma, nonzerocols, itermax, V, w, alpha, tol, batch)
    [N,~] = size(A);
    K = V'*V;
    r = zeros(1,k);
    K0 = K;
    V0 = V;
    iter = 1;
    AV0 = A*V0;
    ck = 0;
    dU = zeros(N,k);
    conv = 0;
    eigvalues = zeros(k,1);
    for iter = 1:itermax
        G = zeros(k-ck,k-ck,N);
        Vj = V(:,ck+1:end);
        dr = zeros(N,k-ck);
        %dUj = dU(:,ck+1:end);
        parfor j = 1:N
            %U = -gamma*4*(A(j, cj)*V0(cj,:) + V0(j,:)*((1-w)*triu(K0) + w*K0));
            U = -gamma*4*(AV0(j,ck+1:end) + V0(j,ck+1:end)*((1-w)*triu(K0(ck+1:end,ck+1:end)) + w*K0(ck+1:end,ck+1:end)));
            G(:,:,j) = (V0(j,ck+1:end))'*U + U'*V0(j,ck+1:end) + U'*U;
            Vj(j,:) = Vj(j,:) + U + alpha*dUj(j,:);
            dr(j,:) = Vj(j,:).^2;
            %dUj(j,:) = U;
            dU(j,ck+1:end) = U;
        end
        %dU(:,ck+1:end) = dUj;
        K(ck+1:end,ck+1:end) = K(ck+1:end,ck+1:end) + sum(G,3);
        r(ck+1:end) = r(ck+1:end) + sum(dr,1);
        Dinv = diag(1./sqrt(r(ck+1:end)));
        K(ck+1:end,ck+1:end) = Dinv*K(ck+1:end,ck+1:end)*Dinv;
        V(:,ck+1:end) = Vj*Dinv;
        Y = A*V(:,ck+1:end);
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
                Y = Y1(:,cplm);
                dU(:,ck+1:end) = zeros(N,k-ck);
            end
        end
        V0(:,ck+1:end) = V(:,ck+1:end);
        K0(ck+1:end,ck+1:end) = K(ck+1:end,ck+1:end);
        AV0(:,ck+1:end) = Y;
        r = zeros(1,k);
    end
    if conv == 0
        [V,~] = qr(V,0);
        H = V'*A*V;
        [Q, D] = eig(H);
        eigvectors = V*Q;
        eigvalues = diag(D);
    end
end
