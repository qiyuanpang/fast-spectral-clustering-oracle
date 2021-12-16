function  [w, mvcput] = user_Hx( v,  Mmat )   
%
% Usage: [w] = user_Hx( v,  Mmat )
%
% compute matrix-vector products.
% the "Mmat" is optional, when it is omitted,
% a global matrix named "A" is necessary.
%
% 
% all matrix-vector products are performed through calls to
% this subroutine so that the mat-vect count can be accurate.
% a global veriable "MVprod" is needed to count the 
% matrix-vector products
%  
% note that if Mmat is a string function, the interface here
% is not complete yet, since in many applications one needs to 
% input more variables to @Mmat than just v
%
  
  global A_operator
  global MVprod       %count the number of matrix-vector products
  global MVcpu        %count the cputime for matrix-vector products
  
  mvcput = cputime;
  if nargin == 1 
     w = A_operator * v;
  else
    if (isnumeric(Mmat))
      w = Mmat * v;
    else
      w = feval(Mmat, v);
    end
  end

  mvcput = cputime - mvcput;  
  %
  % increase the global mat-vect-product count accordingly
  %
  MVcpu  = MVcpu + mvcput;
  MVprod = MVprod + size(v,2);  

  
%end function user_Hx
