%v表示受控比特处于第v个状态时，作用Mat门。v的取值范围是[1,2^(n_qubit-log2(length(Mat)))]
%在出现在受控比特和目标比特之间有无论0,1均执行Mat门，此时v应该是一个数组，
%表示受控比特处于第v()个状态时，作用Mat门。
%高位控制低位
function U = CnU(v, n_qubit,Mat)
U = sparse(2^n_qubit,2^n_qubit);

%构建只执行I门和只执行Mat门的序列，其中v数组是只执行Mat门的序列
%等差数列
w = [1:2^(n_qubit-log2(length(Mat)))];

%剔除v中包含的值
for i =1:2^(n_qubit-log2(length(Mat)))
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
     p_i = sparse(2^(n_qubit-log2(length(Mat))),1);
     p_i(w(i)) = 1;
     U = U + kron(p_i*p_i',sparse(eye(length(Mat))));
end

%把只执行X的加起来
for i =1:length(v)
     p_i = sparse(2^(n_qubit-log2(length(Mat))),1);
     p_i(v(i)) = 1;
     U = U + kron(p_i*p_i',Mat);
end

end