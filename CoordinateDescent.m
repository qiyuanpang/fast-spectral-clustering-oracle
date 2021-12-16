function [eigvalues, eigvectors] = CoordinateDescent(A, k, gamma, nonzerocols, itermax, V)
    [N,~] = size(A);
    %V = randn(N,k);
    K = V'*V;
    r = zeros(1,k);
    K0 = K;
    V0 = V;
    i = 1;
    while i <= itermax
        j = mod(i-1, N) + 1;
        cj = nonzerocols{j};
        U = -gamma*4*(A(j, cj)*V0(cj,:) + V0(j,:)*K0);
        G = (V0(j,:))'*U + U'*V0(j,:) + U'*U;
        V(j,:) = V(j,:) + U;
        r = r + V(j,:).^2;
        K = K + G;
        i = i + 1;
        if j == N
            Dinv = diag(1./sqrt(r));
            K = Dinv*K*Dinv;
            K0 = K;
            V = V*Dinv;
            V0 = V;
            r = zeros(1,k);
        end
    end
    [V,~] = qr(V,0);
    H = V'*A*V;
    [Q, D] = eig(H);
    eigvectors = V*Q;
    eigvalues = diag(D);
end
