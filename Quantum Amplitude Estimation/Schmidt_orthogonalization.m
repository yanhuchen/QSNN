function b=Schmidt_orthogonalization(a)
[m,n] = size(a);
if(m<n)
    error('行小于列，无法计算，请转置后重新输入');
    return
end
b=zeros(m,n);
%正交化
b(:,1)=a(:,1);
for i=2:n
    for j=1:i-1
        b(:,i)=b(:,i)-dot(a(:,i),b(:,j))/dot(b(:,j),b(:,j))*b(:,j);
    end
    b(:,i)=b(:,i)+a(:,i);
end

%单位化
for k=1:n
    b(:,k)=b(:,k)/norm(b(:,k));
end
end