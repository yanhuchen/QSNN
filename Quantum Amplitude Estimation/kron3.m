%三个数按顺序直积
function U = kron3(U1,U2,U3)
U = kron(kron(U1,U2),U3);
end