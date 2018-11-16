 function [Predict_label_f,Score_f,SMR_test_label]=bootstrap_FCM_SVM_n(data,P_L_test_feature,P_L_test_label,P_L_train_feature,P_L_train_label,k,CG_parameter,Num_share)
%FCM Clustering
T=20;m=3;epsm=1.0e-6;
[U,V,objFcn]= myfcm(data, k, T, m, epsm);
Jb=objFcn(end)
maxU=max(U);
% Dividing the results of clustering negative sample data
N_L_train_feature=[];
N_L_test_feature=[];
N_L_test_label=[];
  for i=1:k
      index=find(U(i,:)==maxU);
      N_L_feature=data(index,:);
      N=size(N_L_feature,1);
      rand_N_L=randperm(N);
      rand_N_L_feature=N_L_feature(rand_N_L,:);
      N1=floor(N/5);
      N_L_test_feature=rand_N_L_feature(1:N1,:);
      N_L_test_label=-1*ones(N1,1);
      N_L_train_feature=rand_N_L_feature(N1+1:end,:);
      N_L_train_label=-1*ones(N-N1,1);
      eval(['N_L_train_feature',num2str(i),'=','N_L_train_feature',';']);
      eval(['N_L_test_feature',num2str(i),'=','N_L_test_feature',';']);
      eval(['N_L_test_label',num2str(i),'=','N_L_test_label',';']);
  end
  %合成测试集并统计测试集个数
       N_L_test_feature=[];
       N_L_test_label=[];
   for i=1:k
      eval(['N_L_test_feature=[N_L_test_feature;N_L_test_feature',num2str(i),'];']);
      eval(['N_L_test_label=[N_L_test_label;N_L_test_label',num2str(i),'];']);
   end
     SMR_test_feature=[P_L_test_feature;N_L_test_feature];
     SMR_test_label=[P_L_test_label;N_L_test_label];
     Num_test=size(SMR_test_feature,1);
     
%平均分负类样本数据集     
     Num_feature=[];
     step=[];
for i=1:k
            eval(['Num_feature' num2str(i) '=size(N_L_train_feature' num2str(i) ',1);']);
            eval(['step' num2str(i) '=floor((Num_feature' num2str(i) ')/Num_share);']);
   for j=1:Num_share-1
            eval(['N_L_sub_train_feature' num2str(i) '' num2str(j) '= N_L_train_feature' num2str(i) '((j-1)*step' num2str(i) '+1:j*step' num2str(i) ',:);']);
            eval(['N_L_sub_train_feature' num2str(i) ''  num2str(Num_share) '= N_L_train_feature' num2str(i) '((Num_share-1)*step' num2str(i) '+1:end,:);']);
            eval(['Num_sub_train_feature' num2str(i) '' num2str(j) '=size(N_L_sub_train_feature' num2str(i) '' num2str(j) ',1);']);
            eval(['N_L_sub_train_label' num2str(i) '' num2str(j) '=-1*ones(Num_sub_train_feature' num2str(i) '' num2str(j) ',1);']);
            eval(['Num_sub_train_feature' num2str(i) '' num2str(Num_share) '=size(N_L_sub_train_feature' num2str(i) '' num2str(Num_share) ',1);']);
            eval(['N_L_sub_train_label' num2str(i) '' num2str(Num_share) '=-1*ones(Num_sub_train_feature' num2str(i) '' num2str(Num_share) ',1);']);           
   end
 end
%合并负样本数据集与正样本数据集构成新的训练集
for j=1:Num_share
eval(['N_L_sub_train_feature' num2str(j) '=[];']);
eval(['N_L_sub_train_label' num2str(j) '=[];']);
eval(['SMR_train_feature' num2str(j) '=[];']);
eval(['SMR_train_label' num2str(j) '=[];']);
  for i=1:k
    eval(['N_L_sub_train_feature' num2str(j) '=[N_L_sub_train_feature' num2str(j) ';N_L_sub_train_feature' num2str(i) '' num2str(j) '];']);
    eval(['N_L_sub_train_label' num2str(j) '=[N_L_sub_train_label' num2str(j) ';N_L_sub_train_label' num2str(i) '' num2str(j) '];']);
    eval(['SMR_train_feature' num2str(j) ' =[P_L_train_feature;N_L_sub_train_feature' num2str(j) '];']);
    eval(['SMR_train_label' num2str(j) ' =[P_L_train_label;N_L_sub_train_label' num2str(j) '];']);
  end
end
%训练模型和输出label
for j=1:Num_share
    eval(['model' num2str(j) ' = [];']);
    eval(['model' num2str(j) '=svmtrain(SMR_train_label' num2str(j) ',SMR_train_feature' num2str(j) ',CG_parameter);']);
end
for j=1:Num_share
    eval(['Predict_label_f' num2str(j) ' = [];']);
    eval(['Predict_score_f' num2str(j) ' = [];']);
    eval(['[Predict_label_f' num2str(j) ',accuracy' num2str(j) ',dec_values' num2str(j) ']=svmpredict(SMR_test_label,SMR_test_feature,model' num2str(j) ', '-b 1 ');']);
    eval(['Predict_label_f' num2str(j) '= str2double(Predict_label_f' num2str(j) ');']);
end
%vote
F_Test=[];
for i=1:Num_share
    eval(['F_Test=[F_Test,Predict_label_f' num2str(i) '];']);
end

Predict_label_f = zeros(Num_test,1);
for ii=1:Num_test
	LLII = [];
	LLII = F_Test(ii,:);
	yy_=sum(LLII>0);
	NN_=sum(LLII<0);
	YES_Num(ii) = yy_/Num_share;
	NO_Num(ii) = NN_/Num_share;
	if yy_>NN_
		PP_ = 1;
		Predict_label_f(ii)=PP_;
	else
		PP_ = -1;
		Predict_label_f(ii)=PP_;
	end
end
Score_f = [NO_Num',YES_Num'];
