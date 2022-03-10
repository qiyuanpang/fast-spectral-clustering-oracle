function constructL(D, A)
   #D2inv = sparse(diag(1./sqrt(diag(D))));
   N = size(D,1);
   D2inv = sparse(collect(1:N), collect(1:N), 1 ./broadcast(sqrt,diag(D)), N, N);
   L = sparse(collect(1:N), collect(1:N), ones(N), N, N) - D2inv*A*D2inv;
   return L
end
