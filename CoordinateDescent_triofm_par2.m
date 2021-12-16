function [eigvalues, eigvectors, iter] = CoordinateDescent_triofm_par2(A, k, gamma, nonzerocols, itermax, V, w, alpha)
    [N,~] = size(A);
    K = V'*V;
    r = zeros(1,k);
    K0 = K;
    V0 = V;
    iter = 1;
    dU = zeros(N,k);
    %AV0 = A*V0;
    for iter = 1:itermax
        parfor j = 1:N
            % j = mod(iter-1, N) + 1;
            cj = nonzerocols{j};
            U = -gamma*4*(A(j, cj)*V0(cj,:) + V0(j,:)*((1-w)*triu(K0) + w*K0));
            %U = -gamma*4*(AV0(j,:) + V0(j,:)*((1-w)*triu(K0) + w*K0));
            G = (V0(j,:))'*U + U'*V0(j,:) + U'*U;
            V(j,:) = V(j,:) + U + alpha*dU(j,:);
            r = r + V(j,:).^2;
            K = K + G;
            dU(j,:) = U;
        end
        Dinv = diag(1./sqrt(r));
        K = Dinv*K*Dinv;
        K0 = K;
        V = V*Dinv;
        V0 = V;
        %AV0 = A*V0;
        r = zeros(1,k);
    end
    [V,~] = qr(V,0);
    H = V'*A*V;
    [Q, D] = eig(H);
    eigvectors = V*Q;
    eigvalues = diag(D);
end
