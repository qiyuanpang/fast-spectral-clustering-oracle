function M = sparseSim(str, N, type)
   newstr = strrep(str, '],[', '|');
   newstr = strrep(newstr, '][', '|');
   newstr = strrep(newstr, '[[', '');
   newstr = strrep(newstr, ']]', '');
   data = split(newstr, '|');
   nz = length(data);
   I = zeros(nz,1);
   J = zeros(nz,1);
   V = zeros(nz,1);
   for k = 1:nz
       entry = split(data(k), ',');
       row = str2num(entry{1})+1;
       col = str2num(entry{2})+1;
       val = str2num(entry{3});
       I(k) = row;
       J(k) = col;
       if val > 0
          if strcmp(type, "bin")
             V(k) = 1;
          else
             V(k) = val;
          end
       else
          if strcmp(type, "bin")
             V(k) = 0;
          elseif strcmp(type, "pos")
             V(k) = 0;
          else
             V(k) = val;
          end
       end
   end
   M = sparse(I, J, V, N, N, nz);
end
