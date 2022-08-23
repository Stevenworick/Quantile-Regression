clear;
clc;
global r_
global T
global q
global lambda
global X
global alpha
alpha = 0.85;
lambda=17;
fact_decision = 0.0001;
v_q=[0.99 0.95 0.9 0.8 0.01 0.05 0.1 0.2];

%% Loading data
load('r_brut.mat');
load('header.mat');
date=header(:,1);

%% Statistique du data

des=[mean(r_brut)' std(r_brut)' min(r_brut)' quantile(r_brut,0.05)' quantile(r_brut,0.1)' ...
    quantile(r_brut,0.5)' quantile(r_brut,0.9)' quantile(r_brut,0.95)' max(r_brut)' skewness(r_brut)' kurtosis(r_brut)'];
varnames={'Mean','SD','Min','Value_at_risk5','Value_at_Risk10' ,'Median','Value_at_Risk90','Value_at_Risk95','Max','Skewness','Kurtosis'};
table_des=table(mean(r_brut)',std(r_brut)',min(r_brut)',quantile(r_brut,0.05)',quantile(r_brut,0.1)', ...
    quantile(r_brut,0.5)',quantile(r_brut,0.9)',quantile(r_brut,0.95)',max(r_brut)',skewness(r_brut)',kurtosis(r_brut)','RowNames',header(1,2:end)','VariableNames',varnames);


%% Rendement Standardisé
r=(r_brut-mean(r_brut))./std(r_brut);

%% Calcul Var Hill
vq=[0.01 0.05 0.1 0.2];
var_hill = zeros(8,21);

p=1;
for q = vq
    var = VaR_Hill(r_brut,q);
    var_hill(p,:) = var(:,2);
    var_hill(p+4,:) = var(:,1);
    p=p+1;
end
%% L1_Penalized
eps_opt_all=zeros(21,22);

exitflag_all_L1=zeros(1,21);

A_abs_L1=zeros(21,20);

nb_q_abs_L1=zeros(1,8);
nb_q_opt_sucess=zeros(1,8);

p=1;

for q=v_q
    for i=1:21
        %Rendement à t
        r_=r(2:end,i);

        %Rendement crypto i
        r_i=r(:,i);

        T=size(r_i,1);
        
        %Rendement sauf crypto i
        r_j=[r(:,1:i-1) r(:,i+1:end)];

        %Détermination Quantile Inconditionnel
        var_q_j=quantile(r_j,q);
         
        %Lost Exeedance
        if q < 0.5
            E=r_j.*(r_j<var_q_j);
        else
            E=r_j.*(r_j>var_q_j);
        end
        E=E(2:end,:);

        %Rendement à t-1
        r_i_1=r_i(1:end-1,:);

        %Condition initiale
        X=[ones(size(r_i_1)) E r_i_1];
        bet=inv(X'*X)*(X'*r_);
        
        %optimisation : minimisation de L1 penalized
        options=optimset('MaxIter',5000,'TolFun',10^-12,'TolX',10^-12,'MaxFunEvals',10000);
        [eps_opt,FVAL,EXITFLAG]=fminunc(@L1penalized,bet,options);

        eps_opt_all(i,:)=eps_opt;

        %seuil = absolue? 
        A_abs_L1(i,:)=(abs(eps_opt(2:end-1))>fact_decision)';

        %post lasso (ie) Regression Quantile après selection de coef
        lasso_ind=abs(eps_opt>fact_decision);
        eps_opt_lasso = eps_opt.*(abs(eps_opt)>fact_decision);
        var_lasso=X*eps_opt_lasso;

        ind_lasso=repmat(lasso_ind',1154,1);
        X_lasso = X.*ind_lasso;
        
        %Spillover coefficient
        bet_lasso=inv(X_lasso'*X_lasso)*(X_lasso'*var_lasso);
        exitflag_all_L1(1,i)=EXITFLAG;

        nb2_abs=sum(sum(A_abs_L1));
        nb_opt_sucess=sum((exitflag_all_L1==5));
        bet2=regress(var_lasso,X);
    end
    nb_q_opt_sucess(p)=nb_opt_sucess;
    nb_q_abs_L1(p)=nb2_abs;
    p=p+1;
    
end

nb_q_abs_L1=reshape(nb_q_abs_L1,4,2);

%% L2_penalized
p=1;

exitflag_all_L2=zeros(1,21);

Eps_all_L2 = cell(1,8);
A_abs_L2=zeros(21,20);

nb_q_abs_L2=zeros(1,8);

for q=v_q
    for i=1:21
        %rendement à t
        r_=r(2:end,i);
        
        %rendement crypto i
        r_i=r(:,i);

        T=size(r_i,1);
        
        %Matrice des rendements sauf crypto i
        r_j=[r(:,1:i-1) r(:,i+1:end)];
        
        %Quantile Inconditionel
        var_q_j=quantile(r_j,q);
        
        %Lost Exeedance
        if q < 0.5
            E=r_j.*(r_j<var_q_j);
        else
            E=r_j.*(r_j>var_q_j);
        end
        E=E(2:end,:);

        %Rendement t-1
        r_i_1=r_i(1:end-1,:);

        %Condition Initiale
        X=[ones(size(r_i_1)) E r_i_1];
        bet=inv(X'*X)*(X'*r_);
        
        %Optimisation : minimisation de L2_Penalized
        options=optimset('MaxIter',5000,'TolFun',10^-12,'TolX',10^-12,'MaxFunEvals',10000);
        [eps_opt,FVAL,EXITFLAG]=fminunc(@L2penalized,bet,options);

        eps_opt_all(i,:)=eps_opt;

        %définition matrice A
        A_abs_L2(i,:)=(abs(eps_opt(2:end-1))>fact_decision)';

        %Post Ridge (ie) Regression Quantile après RIdge Selection
        lasso_ind=abs(eps_opt>fact_decision);
        eps_opt_lasso = eps_opt.*(abs(eps_opt)>fact_decision);
        var_lasso=X*eps_opt_lasso;

        ind_lasso=repmat(lasso_ind',1154,1);
        X_lasso = X.*ind_lasso;
        
        %Spillover coefficient
        bet_lasso=inv(X_lasso'*X_lasso)*(X_lasso'*var_lasso);
        exitflag_all_L2(i)=EXITFLAG;

        nb2_abs=sum(sum(A_abs_L2));
        nb_opt_sucess=sum((exitflag_all_L2==5));
        bet2=regress(var_lasso,X);
    end
    Eps_all_L2{p} = eps_opt_all;
    nb_q_abs_L2(p)=nb2_abs;
    p=p+1;
    
end

nb_q_abs_L2=reshape(nb_q_abs_L2,4,2);

%% L2 penalized with 0.05 
nb_q_abs_L2_=zeros(1,8);
for i = 1:8
    nb_q_abs_L2_(i) = sum(sum(Eps_all_L2{1,i}>0.05));
end
nb_q_abs_L2_=reshape(nb_q_abs_L2_,4,2);
%% Elastic_penalized
p=1;

exitflag_all_E=zeros(1,21);

A_abs_E=zeros(21,20);

nb_q_abs_E=zeros(1,8);

for q=v_q
    for i=1:21
        %rendement à t
        r_=r(2:end,i);
        
        %rendement crypto i
        r_i=r(:,i);

        T=size(r_i,1);
        
        %Matrice des rendements sauf crypto i
        r_j=[r(:,1:i-1) r(:,i+1:end)];
        
        %Quantile Inconditionel
        var_q_j=quantile(r_j,q);
%         var_q_j=[var_hill(p,1:i-1) var_hill(p,i+1:end)];
        
        %Lost Exeedance E
        if q < 0.5
            E=r_j.*(r_j<var_q_j);
        else
            E=r_j.*(r_j>var_q_j);
        end
        E=E(2:end,:);

        %Rendement t-1
        r_i_1=r_i(1:end-1,:);

        %détermination condition initiale
        X=[ones(size(r_i_1)) E r_i_1];
        bet=inv(X'*X)*(X'*r_);
        
        %Optimisation : minimisation du ElasticPenalized
        options=optimset('MaxIter',5000,'TolFun',10^-12,'TolX',10^-12,'MaxFunEvals',10000);
        [eps_opt,FVAL,EXITFLAG]=fminunc(@Elasticpenalized,bet,options);

        eps_opt_all(i,:)=eps_opt;

        %définition matrice A
        A_abs_E(i,:)=(abs(eps_opt(2:end-1))>fact_decision)';

        %Post Elastic (ie) Regression Quantile after Elastic Selection
        lasso_ind=abs(eps_opt>fact_decision);
        eps_opt_lasso = eps_opt.*(abs(eps_opt)>fact_decision);
        var_lasso=X*eps_opt_lasso;

        ind_lasso=repmat(lasso_ind',1154,1);
        X_lasso = X.*ind_lasso;
        
        %Spillover coefficient 
        bet_lasso=inv(X_lasso'*X_lasso)*(X_lasso'*var_lasso);
        exitflag_all_E(i)=EXITFLAG;

        nb2_abs=sum(sum(A_abs_E));
        nb_opt_sucess=sum((exitflag_all_E==5));
        
    end
    nb_q_abs_E(p)=nb2_abs;
    p=p+1;
    
end

%Reshaping
nb_q_abs_E=reshape(nb_q_abs_E,4,2);

%% L1_Penalized : Hill Estimator
p=1;

exitflag_all_H=zeros(1,21);

A_abs_H=zeros(21,20);

nb_q_abs_H=zeros(1,8);

for q=v_q
    for i=1:21
        %rendement à t
        r_=r(2:end,i);
        
        %rendement crypto i
        r_i=r(:,i);

        T=size(r_i,1);
        
        %Matrice des rendements sauf crypto i
        r_j=[r(:,1:i-1) r(:,i+1:end)];
        
        %Utilisation de la var de Hill pour déterminer Lost Exeedance
        var_q_j=[var_hill(p,1:i-1) var_hill(p,i+1:end)];
        
        %Lost Exeedance E
        if q < 0.5
            E=r_j.*(r_j<var_q_j);
        else
            E=r_j.*(r_j>var_q_j);
        end
        E=E(2:end,:);

        %Rendement t-1
        r_i_1=r_i(1:end-1,:);

        %détermination condition initiale
        X=[ones(size(r_i_1)) E r_i_1];
        bet=inv(X'*X)*(X'*r_);
        
        %Optimisation : minimisation du ElasticPenalized
        options=optimset('MaxIter',5000,'TolFun',10^-12,'TolX',10^-12,'MaxFunEvals',10000);
        [eps_opt,FVAL,EXITFLAG]=fminunc(@Elasticpenalized,bet,options);

        eps_opt_all(i,:)=eps_opt;

        %définition matrice A
        A_abs_H(i,:)=(abs(eps_opt(2:end-1))>fact_decision)';

        %Post Elastic (ie) Regression Quantile after Elastic Selection
        lasso_ind=abs(eps_opt>fact_decision);
        eps_opt_lasso = eps_opt.*(abs(eps_opt)>fact_decision);
        var_lasso=X*eps_opt_lasso;

        ind_lasso=repmat(lasso_ind',1154,1);
        X_lasso = X.*ind_lasso;
        
        %Spillover coefficient 
        bet_lasso=inv(X_lasso'*X_lasso)*(X_lasso'*var_lasso);
        exitflag_all_H(i)=EXITFLAG;

        nb2_abs=sum(sum(A_abs_H));
        nb_opt_sucess=sum((exitflag_all_H==5));
        
    end
    nb_q_abs_H(p)=nb2_abs;
    p=p+1;
    
end

%Reshaping
nb_q_abs_H=reshape(nb_q_abs_H,4,2);

%% Affichage résultat

table_recap=table(nb_q_abs_L1,nb_q_abs_L2,nb_q_abs_E,'RowNames',{'1%','5%','10%','20%'},'VariableNames',{'LASSO','RIDGE','ELASTIC_NET'});
table_var_hill=table(mean(var_hill,2),'RowNames',{'99%','95%','90%','80%','20%','10%','5%','1%'},'VariableNames',{'mean_VaR_Hill'});
table_L2_=table(nb_q_abs_L2_,'RowNames',{'1%','5%','10%','20%'},'VariableNames',{'RIDGE_seuil'});
table_Hill=table(nb_q_abs_H,'RowNames',{'1%','5%','10%','20%'},'VariableNames',{'L1_Hill'});

disp(table_L2_)
disp(table_var_hill)
disp(table_Hill)
disp(table_recap)

