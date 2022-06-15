function [weights,ValofRisk_EM]=Best_Case_Portfolio(LogReturns,N_stocks,gamma,eta)
% 
%  
% Function that creates the optimal portfolio, within a pool of assets,
% considering Best Case in the General Case, using the emprical
% formulation of the problem.

% INPUT:
%
%   LogReturns:         vector of log returns of all the stocks admissible 
%   Nstocks:            number of stocks selected for the portfolio composition
%   gamma:              risk adversion coefficient
% 
% OUTPUT:
% 
%   weights:            Vector of optimal weights computed numerically 
%   ValofRisk_EM:       Value of Risk computed with the empirical  weights
%   Expected Return:    Daily expected return
%   Volatility:         Volatility of our portfolio


LogReturns=LogReturns(:,1:N_stocks);
N=size(LogReturns,2);

%% Data Preparation

mu=mean(LogReturns,1)';
sigmahat=cov(LogReturns);
A=ones(1,length(mu))*(sigmahat\mu);    
B=mu'*(inv(sigmahat))*mu;
C=ones(1,length(mu))*(sigmahat\ones(length(mu),1));
D=B*C-A^2;

%% Calibration of Theta and S
rng("default")
Gamma=@(x) (gamma*(1-x(1)*gamma*x(2))+x(1))/(1-x(1)*gamma*x(2))^2;   %(20)        
x0=[-rand(1,1),rand(1,1)];  lb=[-inf,1/C]; ub=[0,1];

% options2 = optimoptions(@lsqnonlin,'MaxFunctionEvaluations',6000,'FiniteDifferenceStepSize',[4e-16,1e-15],'MaxIterations',1e4,'FunctionTolerance',eps,...
%     'OptimalityTolerance',eps,'UseParallel',true); 
options2 = optimoptions(@lsqnonlin,'MaxFunctionEvaluations',6000,'FiniteDifferenceStepSize',[eps^(2/3),eps^(2/3)],'MaxIterations',1e4,'FunctionTolerance',eps,...
    'OptimalityTolerance',eps,'UseParallel',true); 

sis= @(x,eta) [ x(1)/2*x(2)*Gamma(x)+1/2*log(1-x(1)*gamma*x(2)) - eta;    %(31)
                x(2) - 1/C*(D/Gamma(x)^2 +1)];

for i=1:length(eta)
[vals(i,:)] = lsqnonlin(@(x) sis(x,eta(i)),x0,lb,ub,options2);%
end
theta=vals(:,1);
S=vals(:,2);

 plot(eta,theta)
% 
% plot(eta,S)

%% Empirical Implementation
sigmatilda =@(a,theta) inv(inv(sigmahat) - theta*gamma*a*a');
Q=@(a,theta)  1/sqrt(det(sigmahat*(inv(sigmatilda(a,theta))))) * exp( - theta * a'* mu + 0.5* (theta^2)*a'*sigmatilda(a,theta)*a);

x0=1/N_stocks*ones(N,1);    E=[];   b=[];   Eeq=ones(1,N);  beq=1;  lb=[]; ub=[];

weights=zeros(N_stocks,length(eta));

options      = optimoptions("fmincon","EnableFeasibilityMode",true,"MaxFunctionEvaluations",1e5,'MaxIterations',1e5,...
                                 'OptimalityTolerance',1e-10,'ConstraintTolerance',1e-6);%
ValofRisk_EM=zeros(length(theta),1);

for i= 1:length(eta)
    Lag= @(a)    1/theta(i) * log(Q(a,theta(i))) + eta(i)/theta(i); %controllare rapporto penalizzante lagrangiano
   
    [weights(:,i),ValofRisk_EM(i)] = fmincon(Lag,x0,E,b,Eeq,beq,lb,ub,[],options);
    ValofRisk_EM(i)=Lag(weights(:,i));%-eta(i)/theta(i)
end

grid on , hold on 
plot(eta,ValofRisk_EM,'-b');
legend('Empirical formula')
title('Value of Risk')

end 
 