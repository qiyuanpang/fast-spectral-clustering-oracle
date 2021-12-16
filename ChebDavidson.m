function [eigvalues, eigvectors] = ChebDavidson(A, x, m, kwant, kkeep, dimmax, tau, itermax)
    x = x/norm(x);
    V = zeros(size(x,1), 2*kwant);
    V(:,1) = x;
    W = zeros(size(x,1), 2*kwant);
    W(:,1) = A*x;
    mu = x'*W(:,1);
    H = [mu];
    r = W(:,1) - mu*x;
    egval = zeros(kwant, 1);
    if norm(r) <= tau
        kc = 1;
        egval(1) = mu;
        H = [];
    else
        kc = 0;
    end
    %estimate the upper bound of eigenvalues
    upperb = norm(A,1);
    lowerb = (upperb + mu) / 2;
    a0 = lowerb;
    ksub = 1;
    iter = 1;
    while iter <= itermax
        t = chebfilter(A, x, m, lowerb, upperb, a0);
        V(:,ksub+1) = GramSchmidt(t, V(:,1:ksub));
        ksub = ksub + 1;
        kold = ksub;
        W(:,ksub) = A*V(:,ksub);
        H(1:ksub-kc, ksub-kc) = V(:,kc+1:ksub)'*W(:,ksub);
        [Y, D] = eigs(H(1:ksub-kc, 1:ksub-kc), ksub-kc);
        mu = D(1,1);
        if ksub >= dimmax
            ksub = kc + kkeep;
            continue
        end
        V(:,kc+1:ksub) = V(:,kc+1:kold)*Y(:,1:ksub-kc);
        W(:,kc+1:ksub) = W(:,kc+1:kold)*Y(:,1:ksub-kc);
        r = W(:,kc+1) - mu*V(:,kc+1);
        noswap = 0;
        iter = iter + 1;
        if norm(r) <= tau*max(diag(D))
            kc = kc + 1;
            egval(kc) = mu;
            % swap eigenpairs if necesary
            vtmp = V(:,kc);
            for i = kc-1:-1:1
                if mu >= egval(i)
                    break
                else
                    noswap = 1;
                    egval(i+1) = egval(i);
                    egval(i) = mu;
                    V(:,i+1) = V(:,i);
                    V(:,i) = vtmp;
                end
            end
        if kc >= kwant & noswap == 0
            break
        end
        lowerb = median(diag(D));
        if a0 > min(diag(D))
            a0 = min(diag(D));
        end
        x = V(:,kc+1);
        H = D(kc+1:ksub, kc+1:ksub);
    end
    %norm(r)
    eigvalues = egval(1:kc);
    eigvectors = V(:, 1:kc);
end
