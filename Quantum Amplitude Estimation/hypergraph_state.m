clc
close
clear
psi = zeros(16,1);
psi(1) = 1;
H = 1/sqrt(2) * [1,1;1,-1];
Z = [1,0;0,-1];
I = eye(2);
U1 = kron(kron(H,H),kron(H,H));
U2 = kron(kron(Z,Z),kron(Z,I));
U3 = kron3(I,CnU(2,2,Z),I);
U4 = kron(CnU([3,4],3,Z),I);
U5 = kron3(CnU(2,2,Z),I,I);
U6 = kron(CnU(4,3,Z),I);
psi = U6 * U5 * U4 * U3 * U2 * U1 * psi;