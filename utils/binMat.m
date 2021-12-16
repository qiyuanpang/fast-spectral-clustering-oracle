function M = binMat(V)
   [n,m] = size(V);
   M = zeros(n,m);
   for i = 1:n
       for j = 1:m
           if V(i,j) > 0 
              M(i,j) = 1;
           else
              M(i,j) = 0;
           end
       end
   end
end
