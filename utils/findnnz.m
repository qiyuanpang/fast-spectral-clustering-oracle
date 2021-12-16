function nonzerosidx = findnnz(A)
   [n,m] = size(A);
   nonzerosidx = {};
   for i = 1:n
       nonzerosidx{end+1} = [];
   end
   [I,J,V] = find(A);
   N = length(I);
   for i = 1:N
       row = I(i);
       col = J(i);
       nonzerosidx{row}(end+1) = col;
   end
end
