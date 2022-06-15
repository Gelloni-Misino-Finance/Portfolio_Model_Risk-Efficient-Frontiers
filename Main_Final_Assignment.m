% Main Final Project Financial Engineering
% AY2021-2022
% Gelloni Gregorio, Misino Bianca

%% Clearing
clear all
clc 
close all
warning off
format long

%% Data Setting

formatDate = 'dd/MM/yyyy';
Quotes=readtable("EUROSTOCK_FINAL.xlsx");
Capitalization=readtable("Capitalization.xlsx");
[Capitalization,index]=sortrows(Capitalization,2,'descend');
Quotes=[Quotes(:,1) Quotes(:,(index+1)')];
idx=find(Quotes.Date ==datetime('04/01/2010','InputFormat',formatDate));
Returns=table2array(Quotes(idx:end,2:end));
LogReturns=log(Returns(2:end,:)./Returns(1:end-1,:));

%% CS1
N_stocks=[10 20 30 46];
gamma=1;
ValofRisk_AN_1=zeros(4,1);
ValofRisk_EM_1=zeros(4,1);
Expected_Return=zeros(4,1);
Volatility=zeros(4,1);
weights_1=zeros(46,1);
alpha_opt_1=zeros(46,1);

for i=1:length(N_stocks)
[weights_1(1:N_stocks(i),i),alpha_opt_1(1:N_stocks(i),i),ValofRisk_AN_1(i,1),ValofRisk_EM_1(i,1),Expected_Return,Volatility(:,i)]=Markovitz_Portfolio(LogReturns,N_stocks(i),gamma);
end

% commento variazione convergenza in funzione di iterazioni  + controllo
% valore atteso e vol , + plot numero max di iterazioni vs numero di solver
% che convergono per numero di stocks (4 plot)

%% CS2_BEST and WORST GENERAL CASE

eta=linspace(0,0.25,200)';
ValofRisk_AN_W1=zeros(length(eta),4);
ValofRisk_EM_W1=zeros(length(eta),4);
ValofRisk_EM_B1=zeros(length(eta),4);
errorW1=zeros(length(eta),4);

weights_W1_10=zeros(N_stocks(1),length(eta));    weights_W1_20=zeros(N_stocks(2),length(eta));      weights_W1_30=zeros(N_stocks(3),length(eta));   weights_W1_46=zeros(N_stocks(4),length(eta));
alpha_opt_W1_10=zeros(N_stocks(1),length(eta));  alpha_opt_W1_20=zeros(N_stocks(2),length(eta));    alpha_opt_W1_30=zeros(N_stocks(3),length(eta)); alpha_opt_W1_46=zeros(N_stocks(4),length(eta));
weights_B1_10=zeros(N_stocks(1),length(eta));    weights_B1_20=zeros(N_stocks(2),length(eta));      weights_B1_30=zeros(N_stocks(3),length(eta));   weights_B1_46=zeros(N_stocks(4),length(eta));

for i=1:length(N_stocks)
    if i==1
         [weights_W1_10,alpha_opt_W1_10,ValofRisk_AN_W1(:,i),ValofRisk_EM_W1(:,i),errorW1(:,i)]= Worst_Case_Portfolio(LogReturns,N_stocks(i),gamma,eta); %controllare eta/theta lagrangiano 
%          [weights_B1_10,ValofRisk_EM_B1(:,i)]=Best_Case_Portfolio(LogReturns,N_stocks(i),gamma,eta);
    elseif i==2
          [weights_W1_20,alpha_opt_W1_20,ValofRisk_AN_W1(:,i),ValofRisk_EM_W1(:,i),errorW1(:,i)]= Worst_Case_Portfolio(LogReturns,N_stocks(i),gamma,eta); %controllare eta/theta lagrangiano 
%          [weights_B1_20,ValofRisk_EM_B1(:,i)]=Best_Case_Portfolio(LogReturns,N_stocks(i),gamma,eta);
    elseif i==3
          [weights_W1_30,alpha_opt_W1_30,ValofRisk_AN_W1(:,i),ValofRisk_EM_W1(:,i),errorW1(:,i)]= Worst_Case_Portfolio(LogReturns,N_stocks(i),gamma,eta); %controllare eta/theta lagrangiano 
%          [weights_B1_30,ValofRisk_EM_B1(:,i)]=Best_Case_Portfolio(LogReturns,N_stocks(i),gamma,eta);
    elseif i==4
          [weights_W1_46,alpha_opt_W1_46,ValofRisk_AN_W1(:,i),ValofRisk_EM_W1(:,i),errorW1(:,i)]= Worst_Case_Portfolio(LogReturns,N_stocks(i),gamma,eta); %controllare eta/theta lagrangiano 
%          [weights_B1_46,ValofRisk_EM_B1(:,i)]=Best_Case_Portfolio(LogReturns,N_stocks(i),gamma,eta);

    end
         [ValofRisk_EM_B1(:,i)]=DataCleaning(ValofRisk_EM_B1(:,i),eta,-1);

end

ValofRisk_EM_1_PLOT=ValofRisk_EM_1'.*ones(length(eta),1);
ValofRisk_AN_1_PLOT=ValofRisk_AN_1'.*ones(length(eta),1);
for i=1:length(N_stocks)
    figure(i)
     grid on, hold on
     plot(eta,ValofRisk_AN_1_PLOT(:,i),'r',eta,ValofRisk_AN_W1(:,i),'b',eta,ValofRisk_EM_B1(:,i),'g');
     legend('Markovitz','Worst Case Portfolio','Best Case Portfolio')
     title('Comparison of Nominal,Best and Worst Case model N=', num2str(N_stocks(i)))
end

%% CS2_BEST and WORST GENERAL CASE GX

eta=linspace(0,0.25,200);
ValofRisk_AN_W2=zeros(length(eta),4);
ValofRisk_EM_W2=zeros(length(eta),4);
ValofRisk_EM_B2=zeros(length(eta),4);
errorW2=zeros(length(eta),4);

weights_W2_10=zeros(N_stocks(1),length(eta));    weights_W2_20=zeros(N_stocks(2),length(eta));      weights_W2_30=zeros(N_stocks(3),length(eta));   weights_W2_46=zeros(N_stocks(4),length(eta));
alpha_opt_W2_10=zeros(N_stocks(1),length(eta));  alpha_opt_W2_20=zeros(N_stocks(2),length(eta));    alpha_opt_W2_30=zeros(N_stocks(3),length(eta)); alpha_opt_W2_46=zeros(N_stocks(4),length(eta));
weights_B2_10=zeros(N_stocks(1),length(eta));    weights_B2_20=zeros(N_stocks(2),length(eta));      weights_B2_30=zeros(N_stocks(3),length(eta));   weights_B2_46=zeros(N_stocks(4),length(eta));

for i=1:length(N_stocks)
    if i==1
         [weights_W2_10, alpha_opt_W2_10,   ValofRisk_AN_W2(:,i),   ValofRisk_EM_W2(:,i),   errorW2(:,i)]= Worst_Case_PortfolioGX(LogReturns,N_stocks(i),gamma,eta); %controllare eta/theta lagrangiano 
         [weights_B2_10, ValofRisk_EM_B2(:,i)]=Best_Case_PortfolioGX(LogReturns,N_stocks(i),gamma,eta);
    elseif i==2
         [weights_W2_20, alpha_opt_W2_20,   ValofRisk_AN_W2(:,i),   ValofRisk_EM_W2(:,i),   errorW2(:,i)]= Worst_Case_PortfolioGX(LogReturns,N_stocks(i),gamma,eta); %controllare eta/theta lagrangiano 
         [weights_B2_20, ValofRisk_EM_B2(:,i)]=Best_Case_PortfolioGX(LogReturns,N_stocks(i),gamma,eta);
    elseif i==3
         [weights_W2_30, alpha_opt_W2_30,   ValofRisk_AN_W2(:,i),   ValofRisk_EM_W2(:,i),   errorW2(:,i)]= Worst_Case_PortfolioGX(LogReturns,N_stocks(i),gamma,eta); %controllare eta/theta lagrangiano 
         [weights_B2_30, ValofRisk_EM_B2(:,i)]=Best_Case_PortfolioGX(LogReturns,N_stocks(i),gamma,eta);
    elseif i==4
         [weights_W2_46, alpha_opt_W2_46,   ValofRisk_AN_W2(:,i),   ValofRisk_EM_W2(:,i),   errorW2(:,i)]= Worst_Case_PortfolioGX(LogReturns,N_stocks(i),gamma,eta); %controllare eta/theta lagrangiano 
         [weights_B2_46, ValofRisk_EM_B2(:,i)]=Best_Case_PortfolioGX(LogReturns,N_stocks(i),gamma,eta);

    end
         [ValofRisk_EM_B2(:,i)]=DataCleaning(ValofRisk_EM_B2(:,i),eta);

end

for i=1:length(N_stocks)
    figure(i)
     grid on, hold on
     plot(eta,ValofRisk_AN_1_PLOT(:,i),'r',eta,ValofRisk_AN_W2(:,i),'b',eta,ValofRisk_EM_B2(:,i),'g');
     legend('Markovitz','Worst Case Portfolio_Glasserman-Xu','Best Case Portfolio_Glasserman-Xu')
     title('Comparison of Nominal,Best and Worst Case model N=', num2str(N_stocks(i)))
end


 %% CS3
 
Dividends=readtable("EUROSTOVK_DIVIDENDS_FINAL.xlsx");
Dividends=[Dividends(:,1) Dividends(:,(index+1)')];
idx=find(Dividends.Date ==datetime('04/01/2010','InputFormat',formatDate));
Dividends_array=table2array(Dividends(idx:end,2:end));
Returns_Adj=Returns+Dividends_array;
LogReturns_Adj=log(Returns_Adj(2:end,:)./Returns_Adj(1:end-1,:));

[weights_DV,alpha_opt_DV,ValofRisk_AN_DV,ValofRisk_EM_DV,Expected_Return_DV,Volatility_DV]=arrayfun(@(i)...
     Markovitz_Portfolio(LogReturns_Adj,N_stocks(i),gamma),(1:length(N_stocks))','UniformOutput',false);

%COMMENTO VARIAZIONE PERCENTUALE IN RELAZIONE A PESI CS1 


%% CS4

N_stocks=46;gamma=1;
[~,alpha_opt_DV_CS4,~,~,mPTF,sPTF]= Markovitz_Portfolio(LogReturns_Adj,N_stocks,gamma);

%% Q0 Plot the efficient frontier of m as a function of s, for a range of values on m?

mu=mean(LogReturns_Adj,1)'; sigma=cov(LogReturns_Adj) ;

A=ones(1,length(mu))*(sigma\mu);  B=mu'*(inv(sigma))*mu; C=ones(1,length(mu))*(sigma\ones(length(mu),1)); D=B*C-A^2;

s_Q=@(m) 1/C+C/D*(m-A/C).^2; 

[MU,CI_MU]= normfit(alpha_opt_DV_CS4'*LogReturns_Adj');  m_Q0=MU-CI_MU:0.001:MU+CI_MU;

figure(5)
plot(s_Q(m_Q0),m_Q0,'b');
hold on
grid on
plot(sPTF,mPTF,'*r');    xlabel('VAR'); ylabel('E[X]'); title('Efiicient Frontier');
legend('Efficient Frontier', 'Nominal Portfolio')


%% Q1 What is the shape of the frontier of m as a function of s? Write the equation of the frontier of m as a function of s.

eta=( 0: 0.01: 0.3)';
[~,alpha_opt_W_DV,~,~,~,mPTF_W,sPTF_W,theta_DV,S_DV]=Worst_Case_PortfolioGX(LogReturns_Adj,N_stocks,gamma,eta);
[MU,CI_MU]= normfit((alpha_opt_W_DV'*LogReturns_Adj')');  

m_Q0=arrayfun(@(i) linspace(MU(i)-CI_MU(i),MU(i)+CI_MU(i),200),(1:length(theta_DV))','UniformOutput',false);
m_Q0=cell2mat(m_Q0);


surf(s_Q(m_Q0),theta_DV,m_Q0 ,'EdgeColor','b','FaceColor','g')
hold on
plot3(S_DV,theta_DV,MU,'*r')
xlabel('Var'); zlabel('E[X]'); ylabel('theta');
title('Efficient frontier for different values of theta')
hold off


%% Q2 What can we say in this case on the efficient frontier ( ˜m as a function of s) comparing it with the nominal case?

m_Q0=MU-CI_MU:0.001:MU+CI_MU;
eta     = [0.05  0.10];
Gamma=@(theta,gamma,S) gamma/(1-theta*gamma*S);   %(36)
Gamma39=@(theta) (gamma*C+sqrt( (gamma*C)^2+4*gamma*theta*(C-theta*gamma)*D) ) /(2*(C-theta*gamma)); %(39)
alpha_opt =@(theta,gamma,S) A/Gamma(theta,gamma,S) *(sigma\mu)/A+(1-A/Gamma(theta,gamma,S))*(sigma\ones(N_stocks,1))/C;     %(38)
% R= @(theta,gamma,S) 0.5*(Gamma(theta,gamma,S)/gamma-1-log(Gamma(theta,gamma,S)/gamma));                                           %(40)
 
% Find the optimal gamma
% z(1)= GAMMA 
% z(2)= COEFFICIENTE LAGRANGIANO
% z(3)= THETA
% z(4)= S
rng('default')

R= @(k) 0.5*(1/k-1-log(1/k));           %(40)
k=arrayfun(@(i) fzero(@(k)R(k)-eta(i),2),(1:2)');

lb=[0,0,0,1/C];
ub=[inf,inf,inf,1];
x0=randn(1,4);
options=optimoptions(@lsqnonlin,'MaxFunctionEvaluations',1e5);
z=zeros(2,4);

for i=1:length(eta)
z(i,:) = lsqnonlin(@(z) [  z(1)*sigma*alpha_opt(z(3),z(1),z(4)) - mu + z(2)*ones(N_stocks,1)            ;
                           z(1)*alpha_opt(z(3),z(1),z(4))'*sigma*alpha_opt(z(3),z(1),z(4)) - mu'*alpha_opt(z(3),z(1),z(4)) + z(2)                     ;
                           z(4) - 1/C*(D/Gamma39(z(3))^2 + 1)                                                   ;  %(29)
                           k(i) - 1 - z(1).*z(3).*z(4);
                           0.5*(Gamma39(z(3))/z(1)-1-log(Gamma39(z(3))/z(1)))-eta(i)                 ] ,x0,lb,ub,options)       ;
end

gamma =  z(:,1);  %gamma ottimale imponendo derivata e derivata *direzione pari a zero
% lambda = z(:,2);  
theta =  z(:,3);
S=       z(:,4);


s_Q2 = @(m,k) (1/C+C/D*(m-A/C).^2)/k;
alpha_opt_Q2 =arrayfun(@(i)alpha_opt(theta(i),gamma(i),S(i)),(1:2)','UniformOutput',false);
alpha_opt_Q2=reshape(cell2mat(alpha_opt_Q2),46,2);

for i=1:2
[MU(i,1),~]= normfit(alpha_opt_Q2(:,i)'*LogReturns_Adj');  
end

figure(6)
plot(s_Q2(m_Q0,k(1)),m_Q0,'b')
hold on
grid on
plot(s_Q2(m_Q0,k(2)),m_Q0,'g');
legend('Efficient Frontier Eta= 5%','Efficient Frontier Eta=10%')
plot(S(1),MU(1),'*r',S(2),MU(2),'*c');
xlabel('Var');
ylabel('E[X]');
hold off


%% Q3 What about the Best case? Can you plot the efficient frontiers m vs s˜ 

eta     = [0.05  0.10];
Gamma   = @(theta,gamma,S) gamma/(1-theta*gamma*S);   %(36)
Gamma39 = @(theta) (gamma*C+sqrt( (gamma*C)^2+4*gamma*theta*(C-theta*gamma)*D) ) /(2*(C-theta*gamma)); %(39)
alpha_opt = @(theta,gamma,S) A/Gamma(theta,gamma,S) *(sigma\mu)/A+(1-A/Gamma(theta,gamma,S))*(sigma\ones(N_stocks,1))/C;     %(38)
                                  
 
% Find the optimal gamma
% z(1)= GAMMA 
% z(2)= COEFFICIENTE LAGRANGIANO
% z(3)= THETA
% z(4)= S
rng(71)

R= @(k) 0.5*(1/k-1-log(1/k));                    %(40)
k=arrayfun(@(i) fzero(@(k)R(k)-eta(i),2),(1:2)');

lb=[0,0,-inf,1/C];
ub=[inf,inf,0,1];
x0=randn(1,4);
options=optimoptions(@lsqnonlin,'MaxFunctionEvaluations',1e5);
z=zeros(2,4);


for i=1:length(eta)
z(i,:) = lsqnonlin(@(z) [  z(1)*sigma*alpha_opt(z(3),z(1),z(4)) - mu + z(2)*ones(N_stocks,1)                                          ;   % First derivative of the Lagrangian Function
                           z(1)*alpha_opt(z(3),z(1),z(4))'*sigma*alpha_opt(z(3),z(1),z(4)) - mu'*alpha_opt(z(3),z(1),z(4)) + z(2)     ;   % First derivative multiplied by a equal to zero
                           z(4) - 1/C*(D/Gamma(z(1),z(3),z(4))^2 + 1)                                                                 ;   %(29)
                           k(i) - 1 + z(1).*z(3).*z(4);
                           0.5*(Gamma(z(3),z(1),z(4))/z(1)-1-log(Gamma(z(3),z(1),z(4))/z(1)))- eta(i)   ] ,x0,lb,ub,options)           ;   %(40)
end

gamma =  z(:,1);  %gamma ottimale imponendo derivata e derivata *direzione pari a zero
lambda = z(:,2); 
theta =  z(:,3);
S=       z(:,4);
% k= 1 - S.*theta.*gamma;

s_Q2 = @(m,k) (1/C+C/D*(m-A/C).^2)./k;

% Computation of mean and variance of the two Portfolios

for i=1:length(eta)
    
    [weights_BGX(:,i),~]=Best_Case_PortfolioGX(LogReturns,N_stocks,gamma(i),eta(i));
    MU_Q3(i,1)= weights_BGX(:,i)'*mu;
    S_Q3(i,1)= weights_BGX(:,i)'*sigma*weights_BGX(:,i);                 

end

figure(7)
plot(s_Q2(m_Q0,k(1)),m_Q0,'b',s_Q2(m_Q0,k(2)),m_Q0,'g')
hold on
grid on
plot(S_Q3(1),MU_Q3(1),'*r',S_Q3(2),MU_Q3(2),'*c');
legend('Efficient Frontier Eta= 5%','Efficient Frontier Eta=10%')

xlabel('Var');
ylabel('E[X]');
hold off



%% Q4 What is the shape of the frontier of m as a function of s?

eta=( 0: 0.01: 0.3)';

rng(71)
Gamma     = @(theta,gamma,S) (gamma*(1-theta*gamma*S) + theta)/(1-theta*gamma*S);
alpha_opt = @(theta,gamma,S) A/Gamma(theta,gamma,S) *(sigma\mu)/A+(1-A/Gamma(theta,gamma,S))*(sigma\ones(N_stocks,1))/C;     %(28)
s_Q=@(m) 1/C+C/D*(m-A/C).^2; 

% Find the optimal gamma
% z(1)= GAMMA 
% z(2)= COEFFICIENTE LAGRANGIANO
% z(3)= THETA
% z(4)= S

lb=[0,0,0,1/C];
ub=[inf,inf,inf,1];
x0=rand(1,4);
options = optimoptions(@lsqnonlin,'MaxFunctionEvaluations',6000,'FiniteDifferenceStepSize',[eps^(2/3),eps^(2/3),eps^(2/3),eps^(2/3)],'MaxIterations',1e4,'FunctionTolerance',eps,...
    'OptimalityTolerance',eps,'UseParallel',true); 

for i=1:length(eta)
z(i,:) = lsqnonlin(@(z) [  z(1)*alpha_opt(z(3),z(1),z(4))'*sigma*alpha_opt(z(3),z(1),z(4)) - mu'*alpha_opt(z(3),z(1),z(4)) + z(2)*alpha_opt(z(3),z(1),z(4))'*ones(N_stocks,1); % First derivative multiplied by a              ;                            
                           alpha_opt(z(3),z(1),z(4))'*ones(N_stocks,1) - 1;
                           z(1)*sigma*alpha_opt(z(3),z(1),z(4)) - mu + z(2)*ones(N_stocks,1)                              ; % First derivative
                           %z(1)*alphaopt(z(1),z(2))'*sigma*alphaopt(z(1),z(2))-mu'*alphaopt(z(1),z(2))+z(2)              ;                                          ; 
                           %alphaopt(z(1),z(2))- alpha_opt(z(3),z(1),z(4))                                                ;
                           z(4) - 1/C*(D/Gamma(z(3),z(1),z(4))^2 + 1)                                                     ;  %(29)                        
                           z(3)/2*z(4)*Gamma(z(3),z(1),z(4)) + 1/2*log(1-z(3)*z(1)*z(4)) - eta(i)] ,x0,lb,ub,options)     ;          %(35)                                           %(31) 

end

gamma =  z(:,1);  %gamma ottimale imponendo derivata e derivata *direzione pari a zero
% lambda = z(:,2);  
theta =  z(:,3);
S=       z(:,4);
[theta]=DataCleaning(theta,eta);
[S]=DataCleaning(S,eta);
[S]=DataCleaning(S,eta);
figure()
plot(eta,theta)
figure()
plot(eta,S)

alpha_opt_Q4=arrayfun(@(i) alpha_opt(theta(i),gamma(i),S(i)),(1:length(eta))','UniformOutput',false);
alpha_opt_Q4=reshape(cell2mat(alpha_opt_Q4),N_stocks,length(eta));

[MU,CI_MU]= normfit((alpha_opt_Q4'*LogReturns_Adj')');  
[MU]=DataCleaning(MU',eta);
[CI_MU]=DataCleaning(CI_MU',eta);
m_Q=arrayfun(@(i) linspace(MU(i)-CI_MU(i),MU(i)+CI_MU(i),200),(1:length(eta))','UniformOutput',false);
m_Q=cell2mat(m_Q);

% for i=1:length(eta)
% [~,alpha_opt_new(:,i),~,~,~]= Worst_Case_Portfolio(LogReturns,N_stocks,gamma(i),eta(i));
% [MU(i,1),CI_MU(i,1)]= normfit((alpha_opt_Q4(:,i)'*LogReturns_Adj')');  
% 
% m_Q(:,i) = linspace(MU(i,1) - CI_MU(i,1), MU(i,1) + CI_MU(i,1),length(eta));
% 
% MU_Q4(i,1)= alpha_opt_Q4(:,i)'*mu;
% S_Q4(i,1)= alpha_opt_Q4(:,i)'*sigma*alpha_opt_Q4(:,i);
% 
% end


S_Q=s_Q(m_Q);
surf(S_Q(2:end,:), theta(2:end), m_Q(2:end,:), 'EdgeColor','b','FaceColor','g')
hold on
plot3(S_Q4,theta,MU_Q4,'*r')
xlabel('Var'); zlabel('E[X]'); ylabel('theta');
title('Efficient frontier for different values of theta')
hold off



%% Q5 [Facultative]: Consider two values for theta s.t. the ball is of radius Eta = 5% and Eta = 10%.
%Plot the efficient frontier ( ~m as a function of ~s) for the Worst case portfolio, comparing it with
%the nominal case. Consider the same range on values of m considered in Q0.


eta     = [0.05  0.10];

% Calibration of the parameters

rng(71)
Gamma     = @(theta,gamma,S) (gamma*(1-theta*gamma*S) + theta)/(1-theta*gamma*S);
alpha_opt = @(theta,gamma,S) A/Gamma(theta,gamma,S) *(sigma\mu)/A+(1-A/Gamma(theta,gamma,S))*(sigma\ones(N_stocks,1))/C;     %(28)
s_Q       = @(m) 1/C+C/D*(m-A/C).^2;                                                             % Nominal case
%s_Q5      = @(m,S,theta,gamma) (1/C+C/D*(m +(1/gamma)*theta*S*Gamma(theta,gamma,S) -A/C).^2)/S;  % Change of measure

% Find the optimal gamma
% z(1)= GAMMA 
% z(2)= COEFFICIENTE LAGRANGIANO
% z(3)= THETA
% z(4)= S

lb=[0,0,0,1/C];
ub=[inf,inf,inf,1];
x0=randn(1,4);
options=optimoptions(@lsqnonlin,'MaxFunctionEvaluations',1e5);
z=zeros(2,4);

for i=1:length(eta)
z(i,:) = lsqnonlin(@(z) [  z(1)*sigma*alpha_opt(z(3),z(1),z(4)) - mu + z(2)*ones(N_stocks,1)            ;
                           z(1)*alpha_opt(z(3),z(1),z(4))'*sigma*alpha_opt(z(3),z(1),z(4)) - mu'*alpha_opt(z(3),z(1),z(4)) + z(2)*alpha_opt(z(3),z(1),z(4))'*ones(N_stocks,1);               
                           z(4) - 1/C*(D/Gamma(z(3),z(1),z(4))^2 + 1)                                                                       %(31)
                           z(3)/2*z(2)*Gamma(z(3),z(1),z(4))+1/2*log(1-z(3)*z(1)*z(4)) - eta(i);                                            %(29)
                                                                          ] ,x0,lb,ub,options)       ;
end

gamma =  z(:,1);  %gamma ottimale imponendo derivata e derivata *direzione pari a zero
lambda = z(:,2);  
theta =  z(:,3);
S=       z(:,4);

% Computation of the two portfolios
sigmatilda = @(a,theta,gamma) inv(inv(sigma) - theta.*gamma.*a*a');
mutilda    = @(a,theta,gamma) mu - theta*sigmatilda(a,theta,gamma)*a;

fun=@(mtilda,stilda,theta) 1/C+C/D*(mtilda.^2*(theta*stilda)^2+(A/C)^2+2*mtilda.*theta*stilda-2*theta*stilda*A/C-2.*mtilda.*A/C)-stilda/(1-theta*stilda*gamma);
alpha_opt_Nominal=zeros(N_stocks,length(eta));
alpha_opt_Worst=zeros(N_stocks,length(eta));

MU_Q5=zeros(length(eta),1);     CI_MU_Q5=zeros(length(eta),1);
m_Q5=zeros(length(eta),200);    m_Q5tilda=zeros(length(eta),200); s_Q5tilda=zeros(2,200);
for i = 1 : length(eta)
 [~,alpha_opt_Nominal(:,i),~,~,mPTF(i),sPTF(i)]= Markovitz_Portfolio(LogReturns_Adj,N_stocks,gamma(i));

[~,alpha_opt_Worst(:,i),~,~,~]     = Worst_Case_Portfolio(LogReturns_Adj,N_stocks,gamma(i),eta(i));   % WORST CASE OPTIMAL PORTFOLIO?
[MU_Q(i,1),CI_MU_Q(i,1)]         = normfit((alpha_opt_Worst(:,i)'*LogReturns_Adj')');    
m_Q(i,:)                          = linspace(MU_Q(i,1) - CI_MU_Q(i,1), MU_Q(i,1) + CI_MU_Q(i,1),200);  % range of m


[MU_Q5(i,1),CI_MU_Q5(i,1)]         = normfit((alpha_opt_Worst(:,i)'*LogReturns_Adj')');    
m_Q5(i,:)                          = linspace(MU_Q5(i,1) - CI_MU_Q5(i,1), MU_Q5(i,1) + CI_MU_Q5(i,1),200);  % range of m
m_Q5tilda(i,:)                     = m_Q5(i,:) - theta(i)*S(i)/(1-theta(i)*gamma(i)*S(i));          % range of mtilda
s_Q5tilda(i,:)                     = arrayfun(@(j) fsolve(@(stilda) fun(m_Q5tilda(i,j),stilda,theta(i)),0),(1:length(m_Q5(i,:)))' );
end

m_PTF5=zeros(2,1);
s_PTF5=zeros(2,1);

for i = 1 : length(eta)

m_PTF5(i,1)                           = alpha_opt_Worst(:,i)'*mutilda(alpha_opt_Worst(:,i),theta(i),gamma(i));
s_PTF5 (i,1)                        = alpha_opt_Worst(:,i)'*sigmatilda(alpha_opt_Worst(:,i),theta(i),gamma(i))*alpha_opt_Worst(:,i);


figure()
hold on
grid on
plot(s_Q(m_Q(i,:)),m_Q(i,:),'g') % Markovitz griglia di m o mtilda? 
legend('Nominal model');
plot(sPTF(i),mPTF(i),'*r')
hold off
title('Efficient frontier considering eta=',eta(i)) 
xlabel('Var');
ylabel('E[X]');

figure()
hold on
grid on
plot(s_Q5tilda(i,:),m_Q5tilda(i,:),'b')
plot(s_PTF5(i),m_PTF5(i),'*r')
legend('Worst Case');
title('Efficient frontier considering eta=',eta(i)) 
xlabel('Var');
ylabel('E[X]');

end
