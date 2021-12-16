function D = constructD_abs(M)
   [n,m] = size(M);
   D = zeros(n,m);
   for i = 1:n
       D(i,i) = sum(abs(M(i,:)));
   end
end
