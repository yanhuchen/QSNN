%v = 0表示控制比特为0时执行X门
%v = 1表示控制比特为1时执行X门
function U = CX(v)
p_0 = [1;0];
p_1 = [0;1];
X = [0,1;1,0];
if v == 1
    U = kron(p_0*p_0',eye(2)) + kron(p_1*p_1',X);
elseif v == 0
    U = kron(p_0*p_0',X) + kron(p_1*p_1',eye(2));
end

end