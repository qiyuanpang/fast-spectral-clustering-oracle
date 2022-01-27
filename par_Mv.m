function y = par_Mv(M, v, ncores)
   N = size(M,1);
   n = floor(N/ncores);
   y = zeros(size(v));
   parfor j = 1:ncores
      st = (j-1)*n+1;
      ed = j*n;
      if j == ncores
          ed = N;
      end
      ysliced = M(st:ed,:)*v;
      y(j,:) = ysliced(1,:);
   end
end
