%确定|xy>-|yx>中正负号的位置
clc
close
clear

r = rand(4,2);%随机生成两个向量
r = sort(r,2);
res = kron(r(:,1),r(:,2)) - kron(r(:,2),r(:,1));
