function D = constructD(M)
   [n,m] = size(M);
   D = zeros(n,m);
   for i = 1:n
       D(i,i) = sum(M(i,:));
   end
end
