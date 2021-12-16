function [eigvalues, eigvectors] = triofm(A, k, alpha, itermax)
    N = size(A,1);
    V = rand(N, k);
    eigvalues = zeros(1,k);
    iter = 1;
    while iter <= itermax
        j = mod(iter, k) + 1;
        V(:,j) = V(:,j) - alpha*(A*V(:,j)+V(:,1:j)*(V(:,1:j)'*V(:,j)));
        eigvalues(:,j) = mean((A*V(:,j))./V(:,j));
        iter = iter + 1;
    end
    [V, ~] = qr(V,0);
    H = V'*A*V;
    [Q,D] = eigs(H, size(H,1));
    eigvectors = V*Q;
    %eigvectors = V;
end
