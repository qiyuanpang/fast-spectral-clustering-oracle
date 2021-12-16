function M = posMat(V)
   [n,m] = size(V);
   M = zeros(n,m);
   for i = 1:n
       for j = 1:m
           if V(i,j) > 0
              M(i,j) = V(i,j);
           end
       end
   end
end
