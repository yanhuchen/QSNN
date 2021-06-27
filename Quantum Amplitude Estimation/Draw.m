clc
close
clear

D = load('data.txt');
figure(1)
plot(1:2:1000, D(:,1), 'LineWidth',5,'Color',[119,136,153]/255)
hold on
plot(1:2:1000, D(:,2), 'LineWidth',3,'Color', [220,20,60]/255 )

plot(1:2:1000, D(:,3), 'LineWidth',3,'Color',[30,144,255]/255)

plot(1:2:1000, D(:,4), 'LineWidth',3,'Color',[69 139 116]/255)

plot(1:2:1000, D(:,5), 'LineWidth',3,'Color',[255 140 0]/255)
legend('<X|Y>','<X|Y>_4','<X|Y>_6','<X|Y>_8','<X|Y>_{10}')
xlabel('The number of simulations')
ylabel('The vector inner product')
grid on;


Y = load('successful_rate.txt');
y = zeros(6,4);
for j=1:4
    for i = 1:1000
        if Y(i,j)>0.4 && Y(i,j)<=0.5
            y(1,j) = y(1,j) +1;
        elseif Y(i,j)>0.5 && Y(i,j)<=0.6
            y(2,j) = y(2,j) +1;
        elseif Y(i,j)>0.6 && Y(i,j)<=0.7
            y(3,j) = y(3,j) +1;
        elseif Y(i,j)>0.7 && Y(i,j)<=0.8
            y(4,j) = y(4,j) +1;
        elseif Y(i,j)>0.8 && Y(i,j)<=0.9
            y(5,j) = y(5,j) +1;
        elseif Y(i,j)>0.9 && Y(i,j)<=1
            y(6,j) = y(6,j) +1;
        end
    end
end
y = y';
figure(2)
bar(y,0.5,'stack');
axis([0.5,4.5, 0,1000])
legend('p=0.4~0.5','p=0.5~0.6','p=0.6~0.7','p=0.7~0.8','p=0.8~0.9','p=0.9~1.0');
grid on;
set(gca,'FontSize',32,'Fontname', 'Times New Roman');
