function [eigvalues, eigvectors, cl, iter, iter_tl] = CoordinateDescent_sc_par(A, k, gamma, nonzerocols, itermax, V, w, alpha, tol, batch)
    [N,~] = size(A);
    K = V'*V;
    r = zeros(1,k);
    K0 = K;
    V0 = V;
    iter = 1;
    dU = zeros(N,k);
    % AV0 = A*V0;
    G = zeros(k,k,N);
    dr = zeros(N,k);
    conv = 0;
    cl = ones(N,1);
    nodes = [1:N]';
    picks = ones(N,1);
    iter_tl = 0;
    for iter = 1:itermax
        n = length(nodes);
        iter_tl = iter_tl + n;
        parfor j = 1:N
            if picks(j)
                U = -gamma*4*(A(j,:)*V0 + V0(j,:)*((1-w)*triu(K0) + w*K0));
                G(:,:,j) = (V0(j,:))'*U + U'*V0(j,:) + U'*U;
                V(j,:) = V(j,:) + U + alpha*dU(j,:);
                dr(j,:) = V(j,:).^2;
                dU(j,:) = U;
            end
        end
        K = K + sum(G(:,:,nodes),3);
        r = r + sum(dr(nodes,:),1);
        Dinv = diag(1./sqrt(r));
        K = Dinv*K*Dinv;
        K0 = K;
        V = V*Dinv;
        V0 = V;
        % AV0 = A*V0;
        r = zeros(1,k);
        if mod(iter, batch) == 0
            [V1,~] = qr(V,0);
            %cl1 = kmeans(V1(:,k), 2);
            cl1 = sign(V1(:,k));
            a1 = nnz(find(cl == cl1));
            a2 = nnz(find(cl ~= cl1));
            if min(a1,a2)/N < tol
                conv = 1;
                cl = cl1;
                %[V,~] = qr(V,0);
                H = V1'*A*V1;
                [Q, D] = eig(H);
                eigvectors = V1*Q;
                eigvalues = diag(D);
                break;
            elseif a1 > a2
                picks = (cl ~= cl1);
                nodes = find(picks == 1);
                cl = cl1;
            else
                picks = (cl == cl1);
                nodes = find(picks == 1);
                cl = cl1;
            end
                
        end
    end
    if conv == 0
        [V,~] = qr(V,0);
        %cl = kmeans(V(:,k), 2);
        cl = sign(V(:,k));
        H = V'*A*V;
        [Q, D] = eig(H);
        eigvectors = V*Q;
        eigvalues = diag(D);
    end
end
