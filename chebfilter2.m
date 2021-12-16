function y = chebfilter2(A, x, m, a, b, a0, p)
    c = 2*(a+b)/3;
    e = c - a;
    x0 = x;
    p = max(1.5, min(exp((c-a0)/e - 1), p));
    x1 = p/e*(A*x0 - c*x0);
    for i = 2:m
        y = 2*p/e*(A*x1 - c*x1) - x0;
        x0 = x1;
        x1 = y;
    end
end
