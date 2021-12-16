function y = chebfilter(A, x, m, a, b, a0, p)
    e = (b - a) / 2;
    c = (b + a) / 2;
    %p = 1.5;
    sgm = e / (a0 - c);
    sgm1 = sgm;
    y = p*(A*x - c*x)*sgm1 / e;
    for i = 2:m
        sgmnew = 1 / (2/sgm1 - sgm);
        ynew = 2*p*(A*y - c*y)*sgmnew / e - sgm*sgmnew*x;
        x = y;
        y = ynew;
        sgm = sgmnew;
    end
end
