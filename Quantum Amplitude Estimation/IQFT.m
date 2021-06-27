%量子逆傅里叶变换
%只需要输入变量n，表示多大规模的IQFT
function U= IQFT(n)
I = eye(2);
H = 1/sqrt(2) * [1,1;1,-1];
U = eye(2^n);
for i = 1:n
    W1 = kron3(eye(2^(i-1)), H, eye(2^(n-i)) );
    if i == 1
        W2 = eye(2^n);
    else
        %把第二层循环的所有受控门乘在一起
        W2 = eye(2^n);
        for j = 1:i-1
            I_0 = eye(2^(j-1));
            v = [2^(i-j)/2+1 : 2^(i-j)];
            I_1 = CnU(v, i-j+1,  u1( -2*pi/2^(i-j+1) ));
            I_2 = eye(2^(n-i));

            W2 = kron3(I_0, I_1, I_2) * W2;
        end
    end
    U =  W1 * W2 * U; %相乘顺序不可换
end

end

%verify = kron(eye(8), H) * kron(eye(4), CnU(2, 2, u1(-pi/2))) * kron(I, CnU([3,4], 3, u1(-pi/4))) * CnU([5:8], 4, u1(-pi/16)) * kron3(eye(4),H,I) * kron3(I, CnU(2,2,u1(-pi/2)), I) * kron(CnU([3,4],3,u1(-pi/4)), I) * kron3(I,H,eye(4)) * kron(CnU(2,2,u1(-pi/2)), eye(4)) * kron(H,eye(8))