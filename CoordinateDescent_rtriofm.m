function [eigvalues, eigvectors] = CoordinateDescent_rtriofm(A, k, gamma, nonzerocols, itermax, V)
    [N,~] = size(A);
    %V = randn(N,k);
    K = V'*V;
    i = 1;
    while i <= itermax
        j = mod(i-1, N) + 1;
        cj = nonzerocols{j};
        U = -gamma*4*(A(j, cj)*V(cj,:) + V(j,:)*triu(K));
        G = (V(j,:))'*U + U'*V(j,:) + U'*U;
        V(j,:) = V(j,:) + U;
        K = K + G;
        i = i + 1;
        V = V*diag(1./sqrt(sum(V.^2)));
        %if j == N
        %    Dinv = diag(1./sqrt(sum(K.^2)));
        %    K = Dinv*K*Dinv;
        %    V = V*Dinv;
        %end
    end
    [V,~] = qr(V,0);
    H = V'*A*V;
    [Q, D] = eig(H);
    eigvectors = V*Q;
    eigvalues = diag(D);
end
