function L = constructL(D, A)
   %D2inv = sparse(diag(1./sqrt(diag(D))));
   N = size(D,1);
   D2inv = sparse([1:N], [1:N], 1./sqrt(diag(D)), N, N);
   L = speye(size(D,1)) - D2inv*A*D2inv;
end
