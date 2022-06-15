function [weights,ValofRisk_EM]=Best_Case_PortfolioGX(LogReturns,N_stocks,gamma,eta)
%  
% Function that creates the optimal portfolio, within a pool of assets,
% considering Best Case dynamics according to Glasserman's model, both using the emprical and analytical
% formulation of the problem.
% 
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
A=ones(1,length(mu))*(sigmahat\mu);    %CAMBIARE LE LENGTH UNA VOLTA PROVATA N 
B=mu'*(inv(sigmahat))*mu;
C=ones(1,length(mu))*(sigmahat\ones(length(mu),1));
D=B*C-A^2;
Gamma=@(theta,S) gamma./(1-theta.*gamma.*S);                                  %(36)
GammaK=@(K) gamma./(1-K.*gamma);                                              %(36)
Gamma39=@(theta) (gamma*C+sqrt( (gamma*C)^2+4*gamma*theta*(C-theta*gamma)*D) ) /(2*(C-theta*gamma));
%% Calibration of Theta and S

rng("default")

vals=zeros(length(eta),2);
x0=-rand(1,1); lb=[-inf]; ub=[0];
options2 = optimoptions(@lsqnonlin,'MaxFunctionEvaluations',1e5,'MaxIterations',1e4,...
   'UseParallel',true); %,'FiniteDifferenceStepSize',[3*eps,3*eps] 'OptimalityTolerance',1e-8,'FunctionTolerance',1e-8
%% Calibration of K = theta*S

 sis1= @(x,eta) [  0.5*x*GammaK(x) + 0.5* log(1-x*gamma) - eta ];      %(35)  
         
 
ms = MultiStart('FunctionTolerance',2e-4,'XTolerance',5e-3,...
    'StartPointsToRun','bounds-ineqs','UseParallel', true);


for i=1:length(eta)

    problem=createOptimProblem('lsqnonlin','objective', @(x) sis1(x,eta(i)),'x0', x0,'lb', lb,'ub', ub,'options',options2);
    [K(i,1)] = run(ms,problem,200);

end

%% Calibration of S and theta separately

sis2= @(x,K,eta) [      %x(2) - 1/C*(D/Gamma(x(1),x(2))^2 +1);                                       %(29)
                        0.5* x*Gamma(x(1),x(2)) + 0.5* log(1- x*gamma) - eta ;       %(35) 
                        K  -    x(1)*x(2)  ]   ;


x0=[-1;0.0042];  lb=[-inf,0]; ub=[0,1];


for i=1:length(eta)

    problem=createOptimProblem('lsqnonlin','objective', @(x) sis2(x,K(i),eta(i)),'x0', x0,'lb', lb,'ub', ub);
    [vals(i,:)] = run(ms,problem,200);

end

theta=vals(:,1);
S=vals(:,2);
% plot(eta,theta)
% plot(eta,S)

% for i=1:length(eta)
% [vals(i,:)] = lsqnonlin(@(x) sis(x,eta(i)),x0,lb,ub,options2);%
% 
% end

% [theta,S]=DataCleaning(theta,S,eta);

% figure (2)
% grid on, hold on 
% plot(theta,Gamma(theta,S))
% plot(theta,Gamma39(theta))

%%  Empirical formulation of the problem
rng('default')

x0=-randn(N,1);    E=[];   b=[];   
Eeq=ones(1,N);  beq=1;  lb=[]; ub=[];

weights      = zeros(N_stocks,length(eta));
ValofRisk_EM = zeros(length(theta),1);
options      = optimoptions("fmincon","EnableFeasibilityMode",true,"MaxFunctionEvaluations",1e7,'MaxIterations',1e7,...
                                 'OptimalityTolerance',1e-10,'ConstraintTolerance',1e-6);
 for i = 1:length(eta)

     Lag=@(a) -(1/(2*theta(i)))*log(1-theta(i)*gamma*a'*sigmahat*a)-(a'*mu)+eta(i)/theta(i);
	 [weights(:,i),~]    = fmincon(Lag,x0,E,b,Eeq,beq,lb,ub,[],options);
     ValofRisk_EM(i)=Lag(weights(:,i));% - eta(i)/theta(i)
    
 end

%  for i=1:length(eta)
%      S_EM(i,1)=weights(:,i)'*sigmahat*weights(:,i);
%  end
%  
% figure()
% grid on , 
% plot(eta,ValofRisk_EM,'-b');
% hold on 
% legend('Analytical formula','Empirical formula')
% title('Value of Risk')
end