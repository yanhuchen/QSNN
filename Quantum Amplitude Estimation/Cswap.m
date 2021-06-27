%该函数描述了只有一个控制比特，一个交换门的情况
%n_qubit 表示受控交换门设计的量子比特数量，因为要交换的比特有可能不想邻
%s1,s2表示交换比特所在的位置
%1是最高位，n_qubit是最低位
function U = Cswap(n_qubit,s1,s2)
%首先构建一个两个待交换的比特可能不想邻的swap门
%根据swap门与CNOT的关系，即一个swap可以拆为3个CNOT
%求出v，即在哪些量子状态下会作用X门
%假设s1<s2 
U  =  zeros(2^n_qubit,2^n_qubit);
for s = 1:2^(s2-s1-1)
    v(s) = 2^(s2-s1-1) + s;
end
W1 = CnX(v,s2-s1+1);

%对于低位是控制比特，高位是目标比特，相较于高位是控制比特，低位是目标比特的情况
%时把v变为2进制后，前后翻转即可

bin_v = dec2bin(v-1);
for i = 1:length(v)
    v2(i) = bin2dec(reverse(bin_v(i,:)));
end
v2 = v2+1;

W2 = XnC(v2,s2-s1+1);
swap = W1*W2*W1;

%接着考虑唯一的一个受控比特和swap门的距离，中间可能出现不想邻的情况
%主要是默认受控比特位置是1，s1，s2在低位
%还有就是swap门涉及的qubit的数量：2^(s2-s1+1)是swap门的规模
for s = 1:2^(s1-1-1)
    v3(s) = 2^(s1-1-1) + s;
end
U = CnU(v3, n_qubit,swap);
end