%该函数的功能实现一次Grover迭代
%可以确定的是，oracle是标记辅助比特为1的所有状态
%因此只需要传入制备初态的算子A
%3表示，这是一个3qubit的Grover迭代
function G = Grover3(A)
Z = [1,0;0,-1];
I = eye(2);
X = [0,1;1,0];

%oracle
%把anc=1的状态全部标记
O = kron3(Z,I,I);
%I_0
%只对全0态做一个反转相位
I_0 = kron3(I,I,X) * CnU(1,3,Z) * kron3(I,I,X);

G = -A * I_0 * A^(-1) * O;
end