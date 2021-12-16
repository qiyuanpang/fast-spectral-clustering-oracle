function y = GramSchmidt(v, base)
    [m,n] = size(base);
    x = zeros(size(v));
    for i = 1:n
        x = x + (v'*base(:,i))/norm(base(:,i),2)*base(:, i);
    end
    y = v - x;
    y = y/norm(y);
end
