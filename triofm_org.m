function [eigvalues, eigvectors] = triofm(A, k, alpha, itermax)
    N = size(A,1);
    V = rand(N, k);
    eigvalues = zeros(1,k);
    iter = 1;
    while iter <= itermax
        V = V - alpha*(A*V + V*triu(V'*V));
        eigvalues = mean((A*V)./V);
        iter = iter + 1;
    end
    [V, ~] = qr(V,0);
    H = V'*A*V;
    [Q,D] = eigs(H, size(H,1));
    eigvectors = V*Q;
    %eigvectors = V;
end
