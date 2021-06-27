%尝试两比特的Grover,输入Grover的初态是Grover算子的其中一个特征向量
clc
close
clear
psi_G = 1/2*[1;1;-1i;-1i];
H = 1/sqrt(2) * [1,1;1,-1];
I = eye(2);
X = [0,1;1,0];
Z = [1,0;0,-1];

G = -kron(H,H) * kron(I,X) * CnU(1,2,Z) * kron(I,X) * kron(H,H) * CnU(2,2,Z);

psi_E = 1/2 * ones(4,1);

psi = kron(psi_E,psi_G);
U1 = kron(I,CnU(2,3,G));
U2 = CnU([3,4],4,G*G);
U3 = kron(IQFT(2),eye(4));

psi =U3 * U2 * U1 * psi
%与手算结果相同
