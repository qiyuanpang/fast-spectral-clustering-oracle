function y = chebfilter2(A, x, m, a, b, a0, p)
    c = (a+b)/2;
    e = c - a;
    x0 = x;
    %p = max(1.5, min(exp((c-a0)/e - 1), p));
    d = 2*e/(c-a0);
    x1 = d/e*(A*x0 - c*x0);
    for i = 2:m
        y = 2*d/e*(A*x1 - c*x1) - x0;
        x0 = x1;
        x1 = y;
    end
end
