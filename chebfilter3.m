function y = chebfilter3(A, x, m, a, b, a0, p)
    e = (b - a) / 2;
    c = (b + a) / 2;
    x0 = x;
    p = max(1.5, min(exp((c-a0)/e - 1), p));
    [N,M] = size(A);
    newA = (A-c*speye(N,M))*p/e;
    [I1,J1] = find(abs(newA) <= 1);
    [I2,J2] = find(newA > 1);
    [I3,J3] = find(newA < -1);
    [N1] = find(abs(newA) <= 1);
    [N2] = find(newA > 1);
    [N3] = find(newA < -1);
    V1 = newA(N1);
    V1 = cos(m*acos(V1(:)));
    V2 = newA(N2);
    V2 = cosh(m*acosh(V2(:)));
    V3 = newA(N3);
    V3 = (-1)^m*cosh(m*acosh(V3(:)));
    newA = sparse([I1;I2;I3], [J1;J2;J3], [V1;V2;V3], N, M);
    y = newA*x;
end
