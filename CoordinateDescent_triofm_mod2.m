function [eigvalues, eigvectors, iter, a] = CoordinateDescent_triofm_mod2(A, k, gamma, nonzerocols, itermax, V, w, alpha, m, a1, b, a0, p, th)
    [N,~] = size(A);
    K = V'*V;
    r = zeros(1,k);
    K0 = K;
    V0 = V;
    iter = 1;
    dU = zeros(N,k);
    a = a1;
    AV0 = A*V0;
    %AV0 = chebfilter2(A, V0, m, a, b, a0, p);
    while iter <= itermax
        j = mod(iter-1, N) + 1;
        cj = nonzerocols{j};
        %U = -gamma*4*(A(j, cj)*V0(cj,:) + V0(j,:)*((1-w)*triu(K0) + w*K0));
        t = rand();
        if t < th
            U = -gamma*4*(AV0(j,:) + V0(j,:)*((1-w)*triu(K) + w*K));
        else
            U = -gamma*4*(AV0(j,:) + V0(j,:)*((1-w)*triu(K0) + w*K0));
        end
        G = (V0(j,:))'*U + U'*V0(j,:) + U'*U;
        V(j,:) = V(j,:) + U + alpha*dU(j,:);
        r = r + V(j,:).^2;
        K = K + G;
        iter = iter + 1;
        dU(j,:) = U;
        if j == N
            Dinv = diag(1./sqrt(r));
            K = (Dinv*K)*Dinv;
            K0 = K;
            V = V*Dinv;
            V0 = V;
            %AV0 = chebfilter2(A, V0, m, a, b, a0, p);
            AV0 = A*V0;
            % %scale = mean(std((A*V)./V));
            r = zeros(1,k);

            % [Q,R] = qr(V,0);
            %H = V'*(A*V);
            %[Q, D] = eig(H);
            %newa = median(diag(D));
            %if newa < b && newa >= a0
            %    a = newa
            %end
            %V = V*Q;
            %V0 = V;
            %K = (Q'*K)*Q;
            %K0 = K;
            %r = zeros(1,k);
        end
    end
    [V,~] = qr(V,0);
    %H = V'*chebfilter2(A, V, m, a, b, a0, p);
    H = V'*(A*V);
    [Q, D] = eig(H);
    eigvectors = V*Q;
    eigvalues = diag(D);
end
