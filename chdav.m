function [eval, eigV, nconv, history] = chdav(varargin)
%
% chdav.m implements the polynomial filtered davidson-type method for 
% computing the smallest eigenpairs of symmetric/Hermitian problems.
%
% [currently the Chebyshev filtering is implemented, other filters
%  will be added later]
%
% Usage:
%      [eval, eigV, nconv, history] = chdav(varargin)
%
% where   chdav(varargin)  is one of the following:
%------------------------------------------------------------------------    
% 1.  chdav(A);
% 2.  chdav(A, nwant);
% 3.  chdav(A, nwant, opts);
% 4.  chdav(Astring, dim, nwant, opts);  %when A is a script function,
%                                        %need to specify the dimension
%------------------------------------------------------------------------
% 
% A is the matrix for the eigenproblem, (dim=size(A,1)), 
%   (it is possible to input only a function for mv products, in that case,
%    the upper bound of eigenvalues should be provided through opts).
%
% nwant is the number of wanted eigen pairs,
%   by default: nwant=5.
%
% opts is a structure containing additional parameters, 
% the accepted field names are:  (the order of field names does not matter)
%
%      polym -- the degree of the Chebyshev polynomial; 
%               by default:  polym=10.
%     filter -- filter # to be used;
%               currently only Chebshev filter is implemented,
%               by default: filter=1.
%        tol -- convergence tolerance for the residual norm of each eigenpair;
%               by default:  tol =1e-8.
%       vmax -- maximum subspace dimension;
%               by default:  vmax= max(2*nwant, 20).
%      itmax -- maximum iteration number;
%               by default:  itmax= max(dim, 300).
%      nkeep -- number of vectors to keep during restart,
%               by default:  nkeep= nwant.
%               (note:  if n is large and nwant is relatively large, 
%                say nwant>=100, then setting nkeep=ceil(nwant/2) is 
%                faster than setting nkeep=nwant)
%         v0 -- the initial vector;
%               by default: v0 = rand(dim,1).
%      displ -- information display level; 
%               (<=0 --no output; 1--some output; >=2 --more output) 
%               by default: displ=1.
%     chksym -- check the symmetry of the input matrix A.
%               if chksym==1 and A is numeric, then isequal(A,A') is called.
%               the default is not to check symmetry of A. 
%     nomore -- if set to 1 or logical(1), then only nwant eigenpairs are
%               computed. (by default, nomore==logical(0))  
%        upb -- upper bound of eigenvalues of the input matrix A.
%               this upper bound is optional. provide this bound only when 
%               you know a good bound; otherwise, the code will figure it out.
%
%
%========== Output variables:
%
%    eval:  converged eigenvalues (optional).
%
%    eigV:  converged eigenvectors (optional, but since eigenvectors are
%           always computed, not specifying this output does not save cputime).
%
%   nconv:  number of converged eigenvalues (optional).
%
% history:  log information (optional)
%           log the following info at each iteration step:
%
%           history(:,1) -- iteration number (the current iteration step)
%           history(:,2) -- cumulative number of matrix-vector products 
%                           at each iteration
%           history(:,3) -- residual norm at each iteration
%           history(:,4) -- current approximation of the wanted eigenvalues
%
%---------------------------------------------------------------------------
%
% As an example:
%
%    A = delsq(numgrid('D',90));   A = A - 1.6e-2*speye(size(A));
%    k = 10;  v=ones(size(A,1),1);
%    opts = struct('vmax', k+5, 'v0', v, 'displ', 0);
%    [eval, eigV] = chdav(A, k, opts);  
%
% will compute the k smallest eigenpairs of A, using the specified
% values in opts and the defaults for the other unspecified parameters.
%

%  
%---y.k. zhou
%   June, 2005, UMN
%

  %
  % use a global variable 'MVprod' to count the matrix-vector products.
  % this number is automatically incremented whenever the user provided
  % mat-vect-product script 'user_Hx' is called.
  % (by this the mat-vect-product count will be accurate, there is no need 
  % to manually increase the count within this code in case one may miss
  % increasing the count somewhere by accident)
  %
  global MVprod      
  MVprod = 0;   %initialize mat-vect-product count to zero
    
  %
  % Process inputs and do error-checking
  %
  
  % if no input arguments, return help.
  if nargin == 0, help chdav, return, end
  
  if isnumeric(varargin{1})
    global A_operator
    A_operator = varargin{1}; 
    [dim]=size(A_operator,1);
    if (dim ~= size(A_operator,2)),
      error('The input numeric matrix A must be a square matrix')
    end
    if (dim <=200), 
      warning('small dimension problem, use eig instead')
      [eigV, eval]=eig(full(A_operator));
      eval = diag(eval);
      if (nargout >2), nconv = dim; end
      if (nargout >3), history=[]; end
      return
    end
    Anumeric = 1;
    %if(~isequal(A_operator, A_operator')),
    if (norm(A_operator - A_operator',1)~=0),
      error('your input matrix to chdav.m is not symmetric/Hermitian');
    end
  else
    A_operator = fcnchk(varargin{1});
    Anumeric = 0;
    dim = varargin{2};  % when A is a string, need to explicitly
                        % input the dimension of A
    if (~isnumeric(dim) | ~isreal(dim) | round(dim)~=dim | dim <1)
      error('A valid matrix dimension is required for the input string function')
    end
  end 
  
  %
  % Set default values or apply existing input options:
  %
  if (nargin < 3-Anumeric),
    nwant = min(dim,5);
  else
    nwant = varargin{3-Anumeric};
    if (~isnumeric(nwant) | nwant < 1 | nwant > dim),
      warning('invalid # of wanted eigenpairs input to chdav.m. use default value')
      nwant = min(dim,5);
    end
  end


  %
  % list all the rest default values, they will be overwritten 
  % if provided through structure opts
  %
  filter = 1;
  polym  = 10;
  tol    = 1e-8;
  vmax   = max(2*nwant, 20);
  itmax  = max(dim, 300);
  nkeep  = nwant;
  %nkeep = min(nwant, floor(vmax/2));
  v0     = rand(dim,1);
  displ  = 1;
  nmore  = 4;               %by default, check 4 more eigenpairs (if necessary)
  compute_upb = logical(1); %by default, an upper bound will be computed
  
  if (nargin >= (4-Anumeric))
    opts = varargin{4-Anumeric};
    if ~isa(opts,'struct')
    error('Options must be a structure. (note chdav does not need ''mode'')')
    end
    
    if isfield(opts,'filter'),
      filter=opts.filter;
      if (filter <0 | filter >4),
	warning('invalid filter# input to chdav.m. use default')
	filter=1;
      end
    end
    
    if isfield(opts,'polym'),
      polym = opts.polym;
      if ~isequal(size(polym),[1,1]) | ~isreal(polym) | (polym<=0)
	warning('invalid polym input to chdav.m. use default value')
	polym  = 10;
      end
    end
    
    %--an input upb is no longer needed becaused a bound can be 
    %--easily computed by lancz_bound()
    if (isfield(opts, 'upb')),
       if  ~isequal(size(opts.upb),[1,1]) | ~isreal(opts.upb)
     	  warning('invalid upper bound input to chdav.m, will recompute')
       else
          upb = opts.upb;
          compute_upb = logical(0);
       end
    end

    if isfield(opts,'tol')
      tol = opts.tol;
      if ~isequal(size(tol),[1,1]) | ~isreal(tol) | (tol<=0)
	warning('invalid tol input to chdav.m. use default value')
	tol = 1e-8;
      elseif  (tol < 1e-14)
	%the Davidson-type methods are not for eps accuracy computation.
	warning('tol input to chdav.m is too small. replace by 1e-13')	
	tol = 1e-13;
      end
    end
    
    if isfield(opts,'vmax')
      vmax = opts.vmax;
      if ~isequal(size(vmax),[1,1]) | ~isreal(vmax) | vmax<nwant+5 | vmax>dim
	warning('invalid vmax input to chdav.m. use default value')
	vmax =  max(2*nwant, 20);
      end
    end
    
    if isfield(opts,'nkeep')
      nkeep = opts.nkeep;
      if ~isequal(size(nkeep),[1,1]) | ~isreal(nkeep) | (nkeep<1) | (nkeep>dim)
	warning('invalid nkeep input to chdav.m. use default value')
	nkeep =  nwant;
      end
    end
    
    if isfield(opts,'itmax')
      itmax = opts.itmax;
      if ~isequal(size(itmax),[1,1]) | ~isreal(itmax) | (itmax< 10)
	warning('invalid itmax input to chdav.m. use default value >=100')
	itmax = max(dim, 100);
      end
    end  
    
    if isfield(opts,'v0')
      v0 = opts.v0(:,1);
      if (size(v0,1)~=dim | ~isreal(v0) | norm(v0,1) < 2.2204e-16),
	warning('invalid v0 input to chdav.m. use default value')
	v0 = rand(dim, 1);
      end
    end   
    
    if isfield(opts,'nomore')
      nomore = opts.nomore;  %control if to compute more than nwant eig pairs
      if (nomore ~= 1 & nomore ~=logical(1)),
	nmore = 4;           %use the default value
      else
	nmore = 0;
      end
    end    

    if isfield(opts,'displ')
      displ = opts.displ;
      if (~isequal(size(displ),[1,1])) | (~isreal(displ))
	warning('invalid displ input to chdav.m. use default value 1')
	displ =1;
      end
    end

    if isfield(opts,'chksym')
      if (Anumeric),
	if(~isequal(A_operator, A_operator')),
	  error('input matrix to chdav.m is not symmetric/Hermitian');
	end
      else
	%need a subroutine to check symmetry of A which is provied by 
	%a script for matrix-vector products
	warning('symmetry of a script function is not checked yet');
      end
    end
  end
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % Now start the main algorithm:
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   if (nargout > 3),  
     longlog = 1;  
   else            
     longlog = 0;
   end 
   
   %--Preallocate memory:
   eval = zeros(nwant,1);    
   resnrm=zeros(nwant,1);
   V = zeros(dim, vmax); 
   W = zeros(dim, vmax); 
   Hn= zeros(vmax, vmax);

   %
   % get the very important upper bound. 
   % if compute_upb==0, one has to make sure the input upb is an upper
   % bound, otherwise the convergence can be very slow. it would be safer
   % to just let the lancz_bound() estimate an upper bound.
   %
   if (compute_upb),
     if (Anumeric), 
       upb = norm(A_operator,1);    %onenrm = upb
       upbc= lancz_bound(dim, 4);   %lanznrm= upb
       upb = min(upb, upbc);
     else
       upb = lancz_bound(dim, 4, A_operator);
     end
   end
   
   nit_total = 1;  % init the total iteration number count
   nconv = 0;      % init number of converged eigenvalues
   nconv1 = 1;     % nconv1 stores nconv+1 (the two-index trick)
   
   CHECKALL=1;     % CHECKALL~=1 can be a more robust choice   
   
   [x, rt]= qr(v0, 0); 
   V(:,1) = x;   

   %
   % note that when A is large, passing A explicitly through function variable 
   % is not as efficient as using A as a global variable. The matlab codes 
   % JDQR, JDCG, LOBPCG all used the more efficient global-data approach.
   % so, we also pass A as a global variable when A is numeric. 
   %
   if (Anumeric),
     W(:,1)  = user_Hx( x );
   else
     W(:,1)  = user_Hx( x, A_operator );
   end
   rho =  x' * W(:,1);
   Hn(1,1) = rho;
   r = W(:,1) - Hn(1,1) * x;

   normr = norm(r);
   if (normr < tol) 
     nconv = nconv +1;
     nconv1 = nconv +1;
     eval(nconv)  = rho;
   end
   beta  = max(2.25e-14, abs(rho));
   tolr  = tol*beta;               %set initial tolerance
   if (displ > 2),
     fprintf (' initial rho=%e,  rnorm = %e\n', rho, normr);
   end
   
   %
   % get the estimate lower bound of unwanted ritz values 
   %
   ritz_nwb = (upb+rho)/2;  
   
   n = 1;  %n stores the subspace dimension (note n is NOT dim, dim:=size(A))
   
   if (longlog == 1),
       history(1, 1) = 1;
       history(1, 2) = MVprod;
       history(1, 3) = normr;
       history(1, 4) = rho;
       nlog = 2;
   end      
   
   while (nit_total <= itmax) 
     n1  = n +1;
     nit_total = nit_total +1;
     if (displ > 2),
       fprintf (' lower bound of unwanted spectrum =%e\n', ritz_nwb);
     end 
     
     switch (filter)
      case 1  % the default chebshev filter
         low_nwb = ritz_nwb;
	 if (Anumeric),
	   t = cheb_filter_slim(x, polym, low_nwb, upb);  
           %if (length(eval) < 1), lowb = low_nwb;  else, lowb = eval(1); end
	   %t = cheb_filter(x, polym, low_nwb, upb, lowb);
	 else
	   if (length(eval) < 1), lowb = low_nwb;  else, lowb = eval(1); end
	   t = cheb_filter(x, polym, low_nwb, upb, lowb, A_operator);
	 end
	 
      case 2  % this implements the chebyshev-filter as proposed
              % in a tech report by Yang and Sorensen.
	      % not working as efficiently as case 1
	      % (note that there is an additional parameter delta
	      % which need to be provided by the user for this approach)
	 lowb =  ritz_nwb;     
         t = cheb_filter1( x, polym, lowb, upb, delta);  
        %t = cheb_filter1( x, polym, rho, upb, delta); %slow/no convergence
      
      case 3  
         lowb =  ritz_nwb; 
         t = cheb_filter3( x, polym, lowb, upb, V(:,1:nconv));  
	  
      case 4
         low_nwb = ritz_nwb;
	 if (length(eval) < 1), lowb = low_nwb;  else, lowb = eval(1); end
	 t = cheb_filter_scal(x, polym, low_nwb, upb, lowb);
	 
      %case 5  % least-square polynomial (not working well)
      %          lowb =  ritz_nwb; 
      %          alphat = max(lowb+2e-3, upb/1e+4);
      %          alphat = min(alphat, upb/2);
      %          t = lspoly( x, polym, lowb, upb, alphat); %slow/no convergence

      otherwise
         error('selected filter does not exist in chdav.m')
     end
     
     %
     % make t orthogonal to all vectors in V
     %
     V(:, n1) =  dgks( V(:, 1:n), t );
     
     %
     % do the new matrix-vector product.
     %
     if (Anumeric),
       W(:, n1) =  user_Hx( V(:, n1) );
     else
       W(:, n1) =  user_Hx( V(:, n1), A_operator );  
     end
     
     %
     % compute only the active part (not including the deflated part)
     %
     Hn(nconv1:n1, n1)= V(:, nconv1:n1)'* W(:, n1); 
     Hn(n1, nconv1:n) = Hn(nconv1:n, n1)';  %for symmetric only


     %
     % compute the eigen-decomposition of the rayleigh-quotient matrix
     %
     [Eig_vec, Eig_val] = eig(Hn(nconv1:n1,nconv1:n1));  
     d_eig = diag(Eig_val);
     size_d_eig = n1 - nconv1 +1;
     
     
     %
     %--reorder Ritz pairs according to 'mode'  
     %--(right now the mode is only "SA" or "SR")
     %--the sort is not necessary since eig(H) already does the sort for
     %--symmetric H
     %%[d_eig, indx] = sort(d_eig);
     %%Eig_vec=Eig_vec(:, indx); 
     

     %
     % determine if need to restart (note that only n1 need to be updated)
     %
     if ( n1 >= vmax ),  
       n1 = max([nconv1,  nwant+5,  min(nkeep + nconv, vmax-5)]);
     end
     
     
     % 
     % it is more efficient to do a Rayleigh-Ritz refinement at each step
     % (with this extra work, certain eigenvectors may convergence faster)
     % Moreover, this refinement makes  H = V'AV  diagonal.  
     % Otherwise the latter deflation won't work properly. 
     % (note that all these tricks works for symmetric/hermitian problems,
     %  but not for non-symmetric ones)
     %
     % by placing the restart test above, one can just keep the necessary
     % columns of V and W.
     % (note that at this step, n still stores the old value)
     %
     icount = n1-nconv1+1;
     V(:,nconv1:n1)=V(:,nconv1:n+1)*Eig_vec(:,1:icount);
     W(:,nconv1:n1)=W(:,nconv1:n+1)*Eig_vec(:,1:icount);
     Hn(nconv1:n1,nconv1:n1) = diag(d_eig(1:icount)); 
     
     
     beta1 = max(abs(d_eig));
     %--deflation and restart may make beta1 too small, (the active subspace
     %--dim is small at the end), so use beta1 only when it is larger.     
     if (beta1 > beta),
       beta = beta1; 
       tolr = tol*beta;
     end
     
     
     rho = d_eig(1);
     x = V(:,nconv1);       %% x = x/norm(x);
   
     r = W(:, nconv1)  - rho*x;   
     normr = norm(r);
     if (displ > 2),
       fprintf (' n = %i,  rho=%e,  rnorm = %e\n', n, rho, normr);
     end
     swap = logical(0);

     if (CHECKALL == 1)
       icount = n1 - nconv1+1;   % this count may be too large and can 
                                 % lead to missed eigenvalues
       icount = min(icount, 3);  % opt to check at most 3 Ritz pairs
     else                                            
       icount = 1;  % reduce to check convergence of only one Ritz pair 
                    % at each step this is a safer choice
     end
     
     if (longlog == 1),
       history(nlog, 1) = nit_total;
       history(nlog, 2) = MVprod;
       history(nlog, 3) = normr;
       history(nlog, 4) = rho;
       nlog = nlog+1;
     end
     
     for ii =  1 : icount

       if  ( normr < tolr )
         nconv = nconv +1;
         nconv1 = nconv +1;
	 if (displ > 0),
	   fprintf ('#% i converged in %i steps, ritz(%3i)=%e, rnorm= %6.4e\n', ...
		    nconv,  nit_total, nconv, rho,  normr)
	 end
         eval(nconv) = rho; 
         resnrm(nconv) = normr;
         
         %%--sort converged eigenvalues. 
	 for i = nconv -1: -1 : 1
	   if (rho < eval(i)),
	     swap = logical(1);
	     if (displ > 2),
	       fprintf(' ==> swap %3i  with %3i\n', i+1, i);
	     end
	     eval(i+1)=eval(i);
	     eval(i) = rho;
	     % only assign once the temp vector vtmp
	     if (i == nconv-1),
	       vtmp =  V(:,nconv);
	     end
	     V(:,i+1)= V(:,i);
	     % pre-determine if a further swap is required
	     if (i > 1 & rho < eval(i-1)),
	       continue;
	     else % swap vtmp to the correct place
	       V(:,i)=vtmp;
	       break;
	     end
	   else
	     break
	   end
	 end
	 
	 %%--do not continue if swap happens after nwant+nmore
         if (nconv >= nwant+nmore & swap),
	   swap = logical(0);
         end

         if (nconv >= nwant & (~swap) | nconv >= nwant+3),
	   if (displ > 1),
	     fprintf('  The converged eigenvalues and residual_norms are:\n')
	     for i = 1 : nconv
	       fprintf( '  eigval(%3i) = %11.8e,   resnrm(%3i) = %8.5e \n', ...
			i,  eval(i), i,  resnrm(i))
	     end
	   end
	   % the eigV array can be saved. but in matlab, one can not just
	   % return V(:, 1:nconv) in the output argument list.
	   % the following step should be unnecessary for other languages.
	   eigV = V(:, 1:nconv);
           
           % screen output some info about the parameters
           if (displ > 0), 
               fprintf(' parameters used for the calculation:\n'),
               fprintf(' polym=%i,  vmax=%i, tol=%e\n',...
                       polym, vmax, tol),
               fprintf(' nconv=%i,  #mat-vect-prod=%i,  #iter=%i\n', ...
                       nconv, vmax, nit_total),
           end
           return
         else
           % update rho, r and x to correspond to the next wanted eigenpair
           rho = d_eig(ii+1);
           x = V(:, nconv1);
           r = W(:, nconv1)  - rho *x;
	   normr = norm(r);
	   
	   if (longlog == 1 & normr < tolr),
	     history(nlog, 1) = nit_total;
	     history(nlog, 2) = MVprod;
	     history(nlog, 3) = normr;
	     history(nlog, 4) = rho;
	     nlog = nlog+1;
	   end
         end
       else
	 break; % exit whenever the first non-convergent Ritz pair is reached
       end
     end

     n = n1;    % update the current subspace dimension count
        
     if ( nit_total > itmax),
       %
       % the following should rarely happen unless the problem is
       % extremely difficult (highly clustered eigenvalues)
       % or the vmax is too small
       %
       fprintf('***** itmax=%i, it_total=%i\n', itmax, nit_total)
       warning('***** chdav.m:  Maximum iteration exceeded\n')
       fprintf('***** nwant=%i, nconv=%i, vmax=%i\n', nwant,nconv,vmax)
       warning('***** it could be that your vmax is too small')
       break
     end

     %
     % update  ritz_nwb  to be the median value of d_eig for the 
     % chebyshev polynomial filtering method. 
     % note that d_eig is from the Rayleigh-Ritz refinement step of
     % a Davidson-type mthod. this median choice is an efficient one,
     % but there could likely be other choices.
     % (the convenience in adapting this bound without extra computation
     % shows the reamrkable advantage in integrating the Chebbyshev 
     % filtering in a Davidson-type method)
     %
     
     %ritz_nwb = median(d_eig);
     ritz_nwb = median(d_eig(1:size_d_eig-1));
     
   end




%-------------------------------------------------------------------------------
 function [y] = cheb_filter(x, polm, low, high, leftb, A)
%   
%  [y] = cheb_filter(x, polm, low, high, A) 
%
%  Chebshev iteration, normalized version.
%
%...input variables:
%
%     x -- the input vector to be filtered
%  polm -- polynomial degree 
%   low -- lower bound of the (unwanted) eigenvalues 
%  high -- upper bound of the full spectrum of A
%     A -- the corresponding matrix (optional) 
%
%...output:
%     y -- the output vector
%
%-------------------------------------------------------------
%  the eigenvalues inside interval [low, high] are to be
%  dampened, i.e., [low, high] is mapped into interval [-1,1].
%  where the wanted eigenvalues are mapped outside [-1,1].
%  
  
  e = (high - low)/2;
  center= (high+low)/2;
  %sigma = e/(low - center);     %this means sigma=-1
  sigma = e/(leftb - center);
  sigma1= sigma;
  %sigma1 = e/(leftb - center);  tau = 2/sigma1;
  
  if (nargin < 6),

    y = user_Hx(x);
    y = (y - center*x) * sigma1/e;
  
    for i = 2: polm
      sigma_new = 1 /(2/sigma1 - sigma);
      %sigma_new = 1 /(tau - sigma);
      ynew = user_Hx(y);
      ynew = (ynew - center*y)* 2*sigma_new/e  - sigma*sigma_new*x;
    
      x = y;
      y = ynew;
    
      sigma = sigma_new;
    end
    
  else
    
    y = user_Hx(x, A);
    y = (y - center*x) * sigma1/e;
  
    for i = 2: polm
      sigma_new = 1 /(2/sigma1 - sigma);
      %sigma_new = 1 /(tau - sigma);
      ynew = user_Hx(y, A);
      ynew = (ynew - center*y)* 2*sigma_new/e  - sigma*sigma_new*x;
    
      x = y;
      y = ynew;
    
      sigma = sigma_new;
    end
    
  end
  
  

%-------------------------------------------------------------------------------
function [y] = cheb_filter1(x, polm, low, high, delta)
%   
%  [y] = cheb_filter1(x, polm, low, high, delta) 
%   
%  polm --polynomial degree 
%  low  --lower bound of the full spectrum of A
%  high --upper bound of the full spectrum of A
%  delta--a parameter to determine the lower bound of unwanted eigenvalues 
%   
%--comment:
%
%  This script applies the Chebyshev filter as proposed in
%  the tech report "Accelerationg the Lanczos algorithm via plynomial 
%  spectral transfermation" by D. Sorensen and C. Yang.
%  Numerical experiments show that applying Chebyshev this way
%  in Chebyshev-Davidson is less efficient than applying 
%  the Chebyshev filter we proposed.  One other advantage of our 
%  approach is that we do not need the extra parameter "delta" which
%  is not straightforward to obtain.
%  (we only need to estimate and upper bound, the lower bound is
%  adjusted automatically inside Chebyshev-Davidson without any
%  extra computation, which is an advantage)
%
  
  global A
  
  alpha = 2*(-high + low)/(cosh(acosh(delta)/polm)+1) + high; 
    
  %--chebshev iteration
  
  y  = (2*A*x - (alpha+high)*x)/(high-alpha);
  
  for i = 2: polm
    ynew = (4*A*y - 2*(alpha+high)*y)/(high-alpha) + x;
    x = y;
    y = ynew;
  end



%-------------------------------------------------------------------------------
function [y] = cheb_filter3(x, polm, low, high, eigV)
%   
%  [y] = cheb_filter3(x, polm, low, high, rho, eigV) 
%
%  Chebshev iteration, normalized version.
%
%  polm --polynomial degree 
%  low  --lower bound of the eigenvalues 
%  high --upper bound of the full spectrum
%  rho  --current ritz value
%  eigV --the deflated converged eigenvectors 
%
%  
  
  e = (high - low)/2;
  center= (high+low)/2;
  sigma = e/(low - center);
  sigma1= sigma;
  
  y = user_Hx(x);
  
  %
  % make the new vectors orthogonal to eigV at each step.
  % (numerical results show that the additional projection against
  % eigV is not very useful.)
  %  
  y = (y - center*x- eigV*(eigV'*x)) * sigma1/e;
  
  for i = 2: polm
    sigma_new = 1 /(2/sigma1 - sigma);
    
    ynew = user_Hx(y);
    ynew = (ynew-center*y-eigV*(eigV'*y))*2*sigma_new/e -sigma*sigma_new*x;
    
    x = y;
    y = ynew;
    
    sigma = sigma_new;
  end

  
%-------------------------------------------------------------------------------
function [y] = cheb_filter_scal(x, polm, low, high, leftb, A)
%
% filter with internal scaling (need 3 bounds:  leftb < low < high)
%  
  
  e = (high - low)/2;
  center= (high+low)/2;
  sigma1 = e/(leftb - center);   
  sigma = sigma1;
  tau = 2/sigma1;
  
  if (nargin < 6),

    y = user_Hx(x);
    y = (y - center*x) * (sigma1/e);
  
    for i = 2 : polm
      sigma_new = 1 /(tau - sigma);
      ynew = user_Hx(y);
      ynew = (ynew - center*y)*(2*sigma_new/e) - (sigma*sigma_new)*x;
    
      x = y;
      y = ynew;
    
      sigma = sigma_new;
    end
    
  else
    
    y = user_Hx(x, A);
    y = (y - center*x) * sigma1/e;
  
    for i = 2: polm
      sigma_new = 1 /(tau - sigma);
      ynew = user_Hx(y, A);
      ynew = (ynew - center*y)* 2*sigma_new/e  - sigma*sigma_new*x;
    
      x = y;
      y = ynew;
    
      sigma = sigma_new;
    end
    
  end


%-------------------------------------------------------------------------------
function [y] = cheb_filter_slim(x, polm, low, high)
%   
%  [y] = cheb_filter_slim(x, polm, low, high) 
%
%  Chebshev iteration, normalized version.
%

  e = (high - low)/2;
  center= (high+low)/2;
  
  y = user_Hx(x);
  y = (-y + center*x)/e;
  
  for i = 2: polm
    
    ynew = user_Hx(y);
    ynew = (- ynew + center*y)* 2/e  - x;
    
    x = y;
    y = ynew;

  end
  

%-------------------------------------------------------------------------------
function [v] = dgks(X, v0, orthtest)
% 
% Usage:  v = dgks( X, v0 ).
%
% Apply DGKS correction (two-step MGS) to ortho-normalize v0 against X,
% i.e.   v = (I - XX^T)*v0;  or  v = (I-XX^T)*(I-XX^T)*v0; 
%  and   v = v /norm(v);  
%  
% It is assumed that X is already ortho-normal, this is very important 
% for the projection  P = I - X*X^T  to be orthogonal.
%
% For debugging purpose, a 3rd variable can be provided to test if
% X is ortho-normal or not. when orthtest is 1, test will be performed.
%
%--y.k. zhou
%  UMN, 2004
%
  
  if (nargin==3),
    if (orthtest==1),
      xorth=norm(X'*X - eye(size(X,2)));
      fprintf('   Orthgonality test:  ||X^t*X - I|| = %e\n', xorth)
    end
  end
  
  re_iterate = 1;
  
  while (re_iterate < 3)
%
% normalize v0 for the sake of numerical stability
% (this step can be important in case of badly scaled v0)
%
  nrmv0 = norm(v0);
  if (nrmv0 <= eps),
    fprintf('*** Warning: In dgks.m the input v0 is a zero vector\n')
    fprintf('*** Auto replace v0 by a random vector randn(size(v0))\n')
    v0 = randn(size(v0));
    v0 = v0/norm(v0);
  else
    v0 = v0 / nrmv0;
    nrmv0 = 1;
  end
  
%  
% the first step Gram-Schmidt: v=(I -XX')v0 
%  
  vtmp = X'*v0;
  v = v0 - X*vtmp;    
  nrmv = norm(v);  
  v = v/nrmv;  % always better to normalize (though it can be done later)
  % orth_test1 = norm(v'*X)

%
% the second step Gram-SchmidtS:  v =(I -XX')*v.
% a condition to check if a 2nd correction is necessary is used.
%
% note  tol = ctan(theta), 
% where theta is the angle between  X*vtmp and v
%
  tol = 1.41422;  % the angle is required to be no smaller than pi/4
                  % otherwise, a 2nd step orthogonalization is necessary

  nrmvtmp = norm(vtmp);		  
  if ( nrmvtmp*tol <= nrmv0 ),
    break;
  else
    % need 2nd step projection
    vtmp = X'*v;
    v = v - X*vtmp;   
    nrmv = norm(v); % use this value for the re-orth condition check
    v = v / nrmv;   
    %orth_test2 = norm(v'*X)
    
    if ( norm(vtmp) <= nrmv ), 
      break;
    else
      % need 3rd step projection
      fprintf(' *** Warning: two steps Gram-Schmidt is not enough! \n')
      vtmp = X'*v;
      v = v - X*vtmp;   
      nrmv = norm(v);  
      v = v / nrmv;
      orth_test3 = norm(v'*X);
      fprintf(' *** After 3rd step Gram-Schmidt, orth_test3=%e\n', orth_test3)
      if ( norm(vtmp) <= nrmv ),
	break;
      else 
	% the following normally doesn't happen, 3-step CGS should be enough
	fprintf('**** Error: three steps GS is not enough => \n')
	fprintf('     v0 is in the subspace it is orthogonalizing to. \n') 
	fprintf('**** Replace v0 by a random vector randn(size(v0))\n')
	v0 = randn(size(v0));  
	re_iterate = re_iterate +1;
      end
    end
    
  end
  if (re_iterate > 2),
    fprintf('\n*##* Error: in calling function [v] = dgks(X, v0)\n')
    fprintf('*##* 6 CGS steps failed, can not re-orth v0 against X\n\n');
  end
  end %while


%-------------------------------------------------------------------------------
function  [upperb, T, beta] = lancz_bound (n, k, A, v0)
%
% Usage: upperb  = lancz_bound (n, k, A)
%
% apply k steps Lanczos to get the upper bound of abs(eig(A)).
%  
% Input: 
%        n  --- dimension
%        k  --- (optional) perform k steps of Lanczos
%               if not provided, k =4 (a relatively small k is enough)
%        A  --- (optional) the matrix (or a script name for MV) 
%  
% Output:
%   upperb  --- estimated upper bound of the eigenvalues
%        T  --- tridiagonal matrix that contains eigenvalue estimates
%  
  
%
% although Lanczos requires A=A', using a small k, it seems this
% code can get an upper bound of abs(eig(A)) even for A~=A'.
% (the small k possibly means that this Lanczos is close to Arnoldi)
%
    
   if (nargin < 2), 
     k = 4; 
   else
     k = min(max(k, 4), 20);    %do not go over 20 steps
   end 

   T = zeros(k);
   
   if (nargin > 3),
       v = v0;  %this is ONLY used for testing, a random initial
                %vector should be used in real computation of the
                %upper bound
   else
       %need to use a random vector to get the desired bound
       v = rand(n,1);     
   end
   v = v/norm(v);
   
   tol = 2.5e-16;    %before ||f|| reaches eps, convergence should
                     %have happened, so no need to ask for a small ||f|| 
   if (nargin < 3),
     f     = user_Hx( v );
     alpha = v'*f;
     f     = f - alpha * v; 
     T(1,1)= alpha;
     
     isbreak = 0;
     for j = 2 : k    %run k steps
       beta = norm(f);
       if (beta > tol),
	 v0 = v;  v = f/beta;
	 f  = user_Hx( v );   f = f - v0*beta;
	 alpha = v'*f;
	 f  = f - v*alpha;
	 T(j,j-1) = beta; T(j-1,j) = beta; T(j,j) = alpha;
       else
	 isbreak = 1;
	 break
       end
     end

     if (isbreak ~=1),
        e = eig(T(1:j,1:j));
     else
        e = eig(T(1:j-1,1:j-1));
     end
     if (beta < 1e-2), 
       beta = beta*10; 
     elseif (beta < 1e-1),
       beta = beta*5;
     end
     upperb = max(e) + beta;
     
   else
     f     = user_Hx( v, A );
     alpha = v'*f;
     f     = f - alpha * v; 
     T(1,1)= alpha;
     
     isbreak = 0;
     for j = 2 : k    %run k steps
       beta = norm(f);
       if (beta > tol),
	 v0 = v;  v = f/beta;
	 f  = user_Hx( v, A );   f = f - v0*beta;
	 alpha = v'*f;
	 f  = f - v*alpha;
	 T(j,j-1) = beta; T(j-1,j) = beta; T(j,j) = alpha;
       else
	 isbreak = 1;
	 break
       end
     end

     if (isbreak ~=1),
        e = eig(T(1:j,1:j));
     else
        e = eig(T(1:j-1,1:j-1));
     end

     if (beta < 1e-2), 
       beta = beta*10; 
     elseif (beta < 1e-1),
       beta = beta*5;
     end
     upperb = max(e) + beta;
   end

   
%-------------------------------------------------------------------------------


%-------------------------------------------------------------------------------

%-------------------------------------------------------------------------------

%-------------------------------------------------------------------------------
