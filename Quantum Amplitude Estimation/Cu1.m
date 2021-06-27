%v = 0表示控制比特为0时执行X门
%v = 1表示控制比特为1时执行X门
function U = Cu1(v,lambda)
p_0 = [1;0];
p_1 = [0;1];
u1 = [1,0;0,exp(1i*lambda)];
if v == 1
    U = kron(p_0*p_0',eye(2)) + kron(p_1*p_1',u1);
else
    U = kron(p_0*p_0',u1) + kron(p_1*p_1',eye(2));
end

end