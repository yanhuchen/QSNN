clc
close
clear

H = 1/sqrt(2) * [1,1;1,-1];
p_0 = [1;0];
p_1 = [0;1];
X = [0,1;1,0];
I = eye(2);

% 初态
psi = zeros(2^3,1);
psi(1) = 1;

U1 = kron3(H, Ry(2*pi/3), Ry(2*pi/12));
U2 = Cswap(3,2,3);
U3 = kron3(H,I,I);

psi = U3 * U2 * U1 * psi;

% 我们发现经过一个swap-test电路后的初态中，有一个振幅为负的，这并不影响测量
% 但为了使用Grover算法，必须保证每个振幅是非负的
% 本可以对于初态加绝对值，但是考虑到Grover算法中还要使用制备初态的算子
% 因此，我们使用CCZ门，保证每个振幅非负
% 但是对于普遍的例子来说，究竟哪个振幅是负的还需要再探究
Z = [1,0;0,-1];
U4 = CnU(3,3,Z);
psi = U4* psi;
A = U4 * U3 *U2 *U1;
%剔除计算机精度限制带来的误差
psi = abs(psi);

%oracle
%把anc=1的状态全部标记
U5 = kron3(Z,I,I)
O = U5;

%制备初态算子的逆
U6 = kron3(H, Ry(-2*pi/3), Ry(-2*pi/12)) * Cswap(3,2,3) * kron3(H,I,I) * CnU(3,3,Z);
%I_0
%只对全0态做一个反转相位
U7 = kron3(I,I,X) * CnU(1,3,Z) * kron3(I,I,X);
I_0 = U7;
%制备初态算子
U8 = U4 * U3 * U2 * U1;

%给出一次Grover迭代的结果
psi = U8 * U7 * U6 * U5 * psi

G = U8 * U7 * U6;



