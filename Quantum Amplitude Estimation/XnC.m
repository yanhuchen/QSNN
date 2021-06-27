%v表示受控比特处于第v个状态时，作用X门。v的取值范围是[1,2^(n_qubit-1)]
%在出现在受控比特和目标比特之间有无论0,1均执行X门，此时v应该是一个数组，
%表示受控比特处于第v()个状态时，作用X门。
%2^n_qubit表示量子比特门的维度，其中n_qubit-1位受控比特数量，1个目标比特
%低位控制高位
function U = XnC(v, n_qubit)
U = sparse(2^n_qubit,2^n_qubit);
X = [0,1;1,0];
%构建只执行I门和只执行X门的序列，其中v数组是只执行X门的序列
%等差数列
w = [1:2^(n_qubit-log2(length(X)))];

%剔除v中包含的值
for i =1:2^(n_qubit-1)
    for j =1:length(v)
        if w(i) == v(j)
            %重复的元素赋值为0
            w(i) = 0;
        end
    end
end
%w中删除为0的元素
w(w==0) = [];

%把只执行I的加起来
for i =1:length(w)
     p_i = sparse(2^(n_qubit-1),1);
     p_i(w(i)) = 1;
     U = U + kron(eye(2),p_i*p_i');
end

%把只执行X的加起来
for i =1:length(v)
     p_i = sparse(2^(n_qubit-1),1);
     p_i(v(i)) = 1;
     U = U + kron(X,p_i*p_i');
end
end