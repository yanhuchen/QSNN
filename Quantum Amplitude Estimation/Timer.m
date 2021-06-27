tic
 swap = swap1 * kron3(I,swap2,I) * kron3(eye(2^2),swap3,eye(2^2)) * kron3(eye(2^3),swap4,eye(2^3)) * kron3(eye(2^4), swap5, eye(2^4));
mytimer1=toc;
disp(mytimer1)