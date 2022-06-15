function [weights,alpha_opt,ValofRisk_AN,ValofRisk_EM,error,theta,S]=Worst_Case_PortfolioGX(LogReturns,N_stocks,gamma,eta)

LogReturns=LogReturns(:,1:N_stocks);
N=size(LogReturns,2);
%% Data Preparation
%RIVEDERE CALIBRAZIONE CHIUSA LSQNONLIN 
mu=mean(LogReturns,1)';
sigma=cov(LogReturns);
A=ones(1,length(mu))*(sigma\mu);    %CAMBIARE LE LENGTH UNA VOLTA PROVATA N 
B=mu'*(inv(sigma))*mu;
C=ones(1,length(mu))*(sigma\ones(length(mu),1));
D=B*C-A^2;
Gamma39=@(theta) (gamma*C+sqrt( (gamma*C)^2+4*gamma*theta*(C-theta*gamma)*D) ) /(2*(C-theta*gamma));
%% Calibration of Theta and S
% eta=( 0: 0.001 : 0.3)';

Gamma=@(x) gamma./(1-gamma.*x(1)*x(2));   %(36)   
options=optimoptions("fsolve", 'MaxIterations',1200,'FunctionTolerance',1e-12,'OptimalityTolerance',1e-10,'UseParallel',true,'MaxFunctionEvaluations',5e3);
x0=rand(2,1); 
vals=zeros(length(eta),2);

                                   
for i =1:length(eta)

  fun=  @(x)   [x(2)-1/C*(D/Gamma(x)^2+1)                           ;          %(29) 
                0.5*(Gamma(x)./gamma -1-log(Gamma(x)./gamma))- eta(i)] ;         %(40)     

[vals(i,:)] =  fsolve(fun,x0,options)  ;   

end


theta=vals(:,1);
S=vals(:,2);



%% Empirical Formulation of the problem



x0=zeros(N,1);    E=[];   b=[];   
Eeq=ones(1,N);  beq=1;  lb=[]; ub=[];

weights      = zeros(N_stocks,length(eta));
ValofRisk_EM = zeros(length(theta),1);
options      = optimoptions("fmincon","EnableFeasibilityMode",true,"MaxFunctionEvaluations",1e8,'MaxIterations',1e8,...
                                 'OptimalityTolerance',1e-10,'ConstraintTolerance',1e-6,'UseParallel',true);%
 for i = 1:length(eta)

     Lag=@(a) -(1/(2*theta(i)))*log(1-theta(i)*gamma*a'*sigma*a)-(a'*mu)+eta(i)/theta(i);
	 [weights(:,i),~]    = fmincon(Lag,x0,E,b,Eeq,beq,lb,ub,[],options);
     ValofRisk_EM(i)=Lag(weights(:,i));
        i
  end



%% Analytical Formulation of the problem

alpha_opt    = arrayfun(@(i)  A/Gamma([theta(i) S(i)]) *(sigma\mu)/A+(1-A/Gamma([theta(i) S(i)]))*(sigma\ones(N,1))/C,(1:length(eta))','UniformOutput',false);
alpha_opt    = reshape(cell2mat(alpha_opt),[N_stocks,length(eta)]);
ValofRisk_AN = arrayfun(@(i) 1/(2*C)*(Gamma([theta(i) S(i)])-D/Gamma([theta(i) S(i)]))-A/C,(1:length(eta))');%Gamma([theta(i) S(i)])

Expected_return=zeros(length(eta),1);
Volatility=zeros(length(eta),1);

for i = 1:length(eta)
Expected_return(i) = alpha_opt(:,i)'*mu;
Volatility(i) = alpha_opt(:,i)'*sigma*alpha_opt(:,i); %gamma/2
end 

figure()

plot(eta,ValofRisk_AN,'-r',eta,ValofRisk_EM,'-b');
grid on 
legend('Analytical formula','Empirical formula','Location','southeast')

% hold on 
% title('Value of Risk')
% 
% figure()
 error=abs(ValofRisk_AN-ValofRisk_EM).^2; 
% plot(eta,error)
end