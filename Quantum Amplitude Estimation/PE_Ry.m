clc
close
clear


theta = 2 * pi;

H = 1/sqrt(2) * [1,1;1,-1];
X = [0,1;1,0];
I = eye(2);
Z = [1,0;0,-1];

psi = [1;0;0;0];

%initial
U1 = kron(H,H * Z * u1(pi/2));

%CRy
U2 = CnU(2,2,Ry(theta));

U3 = kron(H,I);

psi = U3 * U2 * U1 * psi;

%验证成功