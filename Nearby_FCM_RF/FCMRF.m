function [Predict_label_f,Predict_score_f,SMR_test_label]=FCMRF(data,P_L_test_feature,P_L_test_label,P_L_train_feature,P_L_train_label,k,nTree)
    T=20;m=4;epsm=1.0e-6;
    [U,V,objFcn]= myfcm(data, k, T, m, epsm);
    Jb=objFcn(end)
    maxU=max(U);
    T1=20;m1=2;epsm1=1.0e-6;
    [U1,V1,objFcn1]= myfcm(P_L_train_feature, k, T1, m1, epsm1);
    Jb1=objFcn1(end)
    % Dividing the results of clustering negative sample data
    model=[];
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
      SMR_train_feature=[P_L_train_feature;N_L_train_feature];
      SMR_train_label=[P_L_train_label;N_L_train_label];
      model=TreeBagger(nTree,SMR_train_feature,SMR_train_label);
      eval(['N_L_train_feature',num2str(i),'=','N_L_train_feature',';']);
      eval(['SMR_train_feature',num2str(i),'=','SMR_train_feature',';']);
      eval(['SMR_train_label',num2str(i),'=','SMR_train_label',';']); 
      eval(['N_L_test_feature',num2str(i),'=','N_L_test_feature',';']);
      eval(['N_L_test_label',num2str(i),'=','N_L_test_label',';']);
      eval(['model',num2str(i),'=','model',';']);
  end
       N_L_test_feature=[];
       N_L_test_label=[];
   for i=1:k
      eval(['N_L_test_feature=[N_L_test_feature;N_L_test_feature',num2str(i),'];']);
      eval(['N_L_test_label=[N_L_test_label;N_L_test_label',num2str(i),'];']);
   end
     SMR_test_feature=[P_L_test_feature;N_L_test_feature];
     SMR_test_label=[P_L_test_label;N_L_test_label];
     %collect classifiers£»
     N2=size(SMR_test_feature,1);
  for i=1:N2
     X=SMR_test_feature(i,:);
  for j=1:k
     Y1=V(j,:);
     d1(i,j)=mydist(X,Y1);
     value(i)=min(min(d1(i,:)));
     value3(i)=max(max(d1(i,:)));
  for p=1:k
        Y2=V1(p,:);
        d2(i,p)=mydist(X,Y2); 
        value2(i)=min(min(d2(i,:)));
    end
   end
  end
     Predict_label_f=[];
     Predict_score_f=[];
   for i=1:N2
     if value2(i)>value(i)  
        col=find(d1(i,:)==value(i));
        L1=col(:,1);
        %col(i)=find(value(i)==d1(i,:));
        %L1=col(:,1);
        eval(['[Predict_label_f' num2str(i) ',Predict_score_f' num2str(i) ']=predict(model' num2str(L1) ',SMR_test_feature(i,:)); ']);
        eval(['Predict_label_f' num2str(i) '= str2double(Predict_label_f' num2str(i) '); ']);
        eval(['Predict_label_f=[Predict_label_f;Predict_label_f',num2str(i),'];']);
        eval(['Predict_score_f=[Predict_score_f;Predict_score_f',num2str(i),'];']);
     else
        AV_value=(value(i)+value3(i))/2;
        col2=find(d1(i,:)>= AV_value);
        %n=60;
        %w=5;
        %delta_d=(value3(i)-value(i))/n;
        %step=value3(i)-w*delta_d;
        L2=col2(:,1);
        eval(['[Predict_label_f' num2str(i) ',Predict_score_f' num2str(i) ']=predict(model' num2str(L2) ',SMR_test_feature(i,:)); ']);
        eval(['Predict_label_f' num2str(i) '= str2double(Predict_label_f' num2str(i) '); ']);
        eval(['Predict_label_f=[Predict_label_f;Predict_label_f',num2str(i),'];']);
        eval(['Predict_score_f=[Predict_score_f;Predict_score_f',num2str(i),'];']);
         
     end
   end
     
     
