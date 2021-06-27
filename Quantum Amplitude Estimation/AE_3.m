clear 
close
clc
%估计比特数为3
%Grover迭代设计的比特数为3

%随机生成一个1024维度的向量，且保证向量的平方和为1
dim = 2;
[x,mul1] = rand_vec(dim);
[y,mul2] = rand_vec(dim);
N = 500;
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

H = 1/sqrt(2) * [1,1;1,-1];
X = [0,1;1,0];
I = eye(2);

%初态
%目标寄存器3个qubit，估计寄存器3个qubit
psi = zeros(2^6,1);
psi(1) = 1;

%U1 = kron3(H, Ry(2*pi/3), Ry(2*pi/12));
U1 = kron3(H, Ux, Uy);%随机生成的Ux，Uy
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
%A = U4 * U3 * U2 * U1;
B = kron(u1(-pi/2),eye(4)) * kron3(H,H,H);
W1 = kron(kron3(H,H,H), A);
%W1 = kron(kron3(H,H,H), B);
psi = W1 * psi;
%受控G门
G = Grover3(A);
W2 = kron3(I, I, CnU(2, 4, G));
W3 = kron(I, CnU([3,4], 5, G^2));
W4= CnU([5,6,7,8], 6, G^4);

psi = W4 * W3 * W2 * psi;

%IQFT
Q = IQFT(3);
swap = CnX([3,4],3) * XnC([2,4],3) * CnX([3,4],3);

psi = kron(swap * Q,eye(8)) * psi;

for i = 1:8
    measure(i) = sum(abs(psi(8*(i-1)+ 1 : 8*(i-1)+ 8)).^2);
end

es(j) = find(measure(1:4) == max(measure(1:4)))-1;
es(j) = es(j) / 8 * pi;
es_res(j) = sqrt(1-2*sin(es(j))^2);
end
plot(1:N,res,1:N,es_res)
hold on
