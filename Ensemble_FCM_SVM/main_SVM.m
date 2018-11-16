clc
clear
load SMR_DCT_feature
% negative data
k=20;Num_share=51;
CG_parameter = '-c 4 -g 0.5 -b 1';
nTree=300;
negative_data1=N_SMR_DCT;
data=negative_data1;
% positive data
P=size(P_SMR_DCT,1);
rand_P_L=randperm(P);
rand_P_L_feature=P_SMR_DCT(rand_P_L,:);
P_L_test_feature=rand_P_L_feature(1:94,:);
P_L_test_label=ones(94,1);
P_L_train_feature=rand_P_L_feature(95:end,:);
P_L_train_label=ones(376,1);
[Predict_label_f,Score_f,SMR_test_label]=bootstrap_FCM_SVM_n(data,P_L_test_feature,P_L_test_label,P_L_train_feature,P_L_train_label,k,CG_parameter,Num_share);

%Results and ROC
[Y1,X1,THRE,AUC1,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(SMR_test_label,Score_f(:,2),'1');
[ACC,SN,Spec,PE,NPV,F_score,MCC] = roc(Predict_label_f,SMR_test_label );
Result=[];
Result = [Result;[ACC,SN,Spec,PE,NPV,F_score,MCC]]
