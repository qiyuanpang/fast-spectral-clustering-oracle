function X = approInverse(L, B, blk_size)
    N = size(L,1);
    k = size(B,2);
    X = zeros(N,k);
    n = ceil(N/blk_size);
    for i = 1:n
        START = (i-1)*blk_size + 1;
        END = min(i*blk_size, N);
        X(START:END, :) = L(START:END, START:END)\B(START:END,:);
    end
    [X, ~] = qr(X, 0);
end
