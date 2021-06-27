clc
close
clear

%估计比特数为4
%Grover迭代设计的比特数为3

H = 1/sqrt(2) * [1,1;1,-1];
X = [0,1;1,0];
H4 = kron(kron3(H,H,H),H);
I = eye(2);
%随机生成一个1024维度的向量，且保证向量的平方和为1
dim = 2;
% [x,mul1] = rand_vec(dim);
% [y,mul2] = rand_vec(dim);
N = 1000;
res = zeros(N,1);
pro = zeros(N,1);
es_res = zeros(N,1);

for j=1:N
    x = [cos((j)/N*pi/4); sin((j)/N*pi/4)];
    y = [sin((j)/N*pi/4);cos((j)/N*pi/4)];
    res(j) = x'*y;
    
%使用施密特正交化对随机向量生成酉矩阵
    C = diag(ones(dim,1));
    C(:,1) = x;
    Ux = Schmidt_orthogonalization(C);
    Ux = Ux';
    D = diag(ones(dim,1));
    D(:,1) = y;
    Uy = Schmidt_orthogonalization(D);
    Uy = Uy';

%初态
%目标寄存器3个qubit，估计寄存器3个qubit
    psi = zeros(2^11,1);
    psi(1) = 1;

    U1 = kron3(H, Ux, Uy);
    U2 = Cswap(3,2,3);
    U3 = kron3(H,I,I);

    %对于执行Grover算子的目标寄存器我们的初态为：
    %我们发现经过一个swap-test电路后的初态中，有一个振幅为负的，这并不影响测量
    %但为了使用Grover算法，必须保证每个振幅是非负的
    %本可以对于初态加绝对值，但是考虑到Grover算法中还要使用制备初态的算子
    %因此，我们使用CCZ门，保证每个振幅非负
    %但是对于普遍的例子来说，究竟哪个振幅是负的还需要再探究
    Z = [1,0;0,-1];
    U4 = CnU(3,3,Z);


    %完整初态
    A = U3 * U2 * U1;

    W1 = kron3(H4,H4, A);

    %受控G门
    G = Grover3(A);
    W2 = kron(eye(2^7), CnU(2, 4, G));
    W3 = kron(eye(2^6), CnU([3,4], 5, G^2));
    W4 = kron(eye(2^5),CnU([5,6,7,8], 6, G^4));
    W5 = kron(eye(2^4),CnU([9:16], 7, G^8));
    W6 = kron(eye(2^3),CnU([17:32], 8, G^16));
    W7 = kron(eye(2^2),CnU([33:64], 9, G^32));
    W8 = kron(eye(2^1),CnU([65:128], 10, G^64));
    W9 = CnU([129:256],11, G^128);
    psi = W9 * W8 * W7 * W6 * W5 * W4 * W3 * W2 * W1 * psi;

    %IQFT
    Q = IQFT(8);
    swap1 = CnX([65:128], 8) * XnC([2:2:128], 8) * CnX([65:128], 8);
    swap2 = CnX([17:32], 6) * XnC([2:2:32], 6) * CnX([17:32], 6);
    swap3 = CnX([5,6,7,8],4) * XnC([2,4,6,8],4) * CnX([5,6,7,8],4);
    swap4 = CnX([1],2) * XnC([1],2) * CnX([1],2);
    swap = swap1 * kron3(I,swap2,I) * kron3(eye(2^2),swap3,eye(2^2)) * kron3(eye(2^3), swap4, eye(2^3));

    psi = kron(swap * Q,eye(8)) * psi;

    for i = 1:2^8
        measure(i) = sum(abs(psi(8*(i-1)+ 1 : 8*(i-1)+ 8)).^2);
    end
    
     M(1) = measure(1);
    for k = 2:2^8/2
        M(k) = measure(k) + measure(2^8+2-k);
    end
    
    %计算内积
    es(j) = find(M == max(M))-1;
    es(j) = es(j) / 2^8 * pi;
    es_res(j) = sqrt(1-2*sin(es(j))^2);
    
    %计算成功率
    pro(j) = max(M);
    
%     %计算成功率
%     inner = 2^8/pi * asin(sqrt(1/2-1/2*res(j)^2)) + 1
%     if floor(inner) ~=1
%         r(j) = (measure(ceil(inner)) + measure(floor(inner))) * 2;
%     else
%         r(j) = measure(ceil(inner))* 2 + measure(floor(inner));
%     end
end
figure(1)
plot(1:N,res,1:N,es_res)
figure(2)
plot(1:N,pro)
