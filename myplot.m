function myplot(U,objFcn)
% ��������U������ӻ�
 
figure(1)
subplot(3,1,1);
plot(U(1,:),'-b');
title('�����Ⱦ���ֵ')
ylabel('��һ��')
subplot(3,1,2);
plot(U(2,:),'-r');
ylabel('�ڶ���')
subplot(3,1,3);
plot(U(3,:),'-g');
xlabel('������')
ylabel('������')
figure(2)
grid on
plot(objFcn);
title('Ŀ�꺯���仯ֵ');
xlabel('��������')
ylabel('Ŀ�꺯��ֵ')
 
 

