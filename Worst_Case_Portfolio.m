function [weights,alpha_opt,ValofRisk_AN,ValofRisk_EM,error]=Worst_Case_Portfolio(LogReturns,N_stocks,gamma,eta)
%  
% Function that creates the optimal portfolio, within a pool of assets, considering Worst Case dynamics, both using the emprical and analytical
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
%   alpha_nominal:      Vector of optimal weights computed analitically 
%   ValofRisk_AN:       Value of Risk computed with the analitical weights
%   ValofRisk_EM:       Value of Risk computed with the empirical  weights
%   Expected Return:    Daily expected return
%   Volatility:         Volatility of our portfolio
%   error:              Squared error between the analytical and empirical formulation of the problem


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

rng(71)
Gamma=@(x) (gamma.*(1-gamma.*x(1)*x(2))+x(1))./(1-gamma.*x(1)*x(2)).^2;   %(20)
options=optimoptions("fsolve",'Algorithm','levenberg-marquardt', 'OptimalityTolerance',1e-10,'MaxFunctionEvaluations',5e3,'MaxIterations',1e4,'FunctionTolerance',1e-10);
x0=rand(2,1); 
vals=zeros(length(eta),2);

for i =1:length(eta)

    fun= @(x) [ x(2) - 1/C*(D/Gamma(x).^2 + 1 );                                     %(29)
                x(1)/2 .*x(2).*Gamma(x) + 1/2*log( 1 - gamma.*x(1).*x(2)) - eta(i)]; %(31)
    [vals(i,:)] =  fsolve(fun,x0,options)  ;   
end


theta=vals(:,1);
S=vals(:,2);
plot(eta,theta)

%% Check Parameters Value Calibration
% theta=zeros(length(eta),1);
% S=zeros(length(eta),1);

% vals=zeros(2,length(eta));
% rng default % For reproducibility 
% 
% for i=1:length(eta)
% problem = createOptimProblem('fmincon','objective', fun, 'x0', x0, ...
%         'Aineq', E, 'bineq', b, 'Aeq', Eeq, 'beq', beq, 'lb', lb, ...
%         'ub', ub,'options',options,'nonlcon',@(x) nonlinearWC(x,eta(i),gamma));%
% 
% ms =MultiStart("StartPointsToRun","all","UseParallel",true);%
% 
% [vals(:,i)] = run(ms,problem,500);
% 
% end
%% Empirical Formulation of the problem

rng('default')
x0=(1/N_stocks).*ones(N,1);    E=[];   b=[];   
Eeq=ones(1,N);  beq=1;  lb=[]; ub=[];

weights=zeros(N_stocks,length(eta));
ValofRisk_EM=zeros(length(theta),1);

for i= 1:length(eta)
    Lag              = @(a) -( 1/(2*theta(i)) )*log( 1-theta(i)*gamma*a'*sigmahat*a )-a'*mu + 1/2 *( theta(i)*a'*sigmahat*a )/( 1 - theta(i)*gamma*a'*sigmahat*a ) +eta(i)/theta(i);%
 	[weights(:,i),~] = fmincon(Lag,x0,E,b,Eeq,beq,lb,ub);
    ValofRisk_EM(i)  = Lag(weights(:,i));
end

% Check to control if the S are well calibrated or not
% for i=1:length(eta)
% AAA(i,1)=weights(:,i)'*sigmahat*weights(:,i);
% end

%% Analytical Formulation of the problem


alpha_opt=arrayfun(@(i) (A/Gamma([theta(i),S(i)])*(sigmahat\mu)/A+(1-A/Gamma([theta(i),S(i)]))*(sigmahat\ones(N,1))/C),(1:length(eta))','UniformOutput',false);
alpha_opt=reshape(cell2mat(alpha_opt),[N_stocks,length(eta)]);

 ValofRisk_AN=arrayfun(@(i) -(1/(2*theta(i)))*log(1-theta(i)*gamma*alpha_opt(:,i)'...
     *sigmahat*alpha_opt(:,i))-alpha_opt(:,i)'*mu+1/2*(theta(i)*alpha_opt(:,i)'...
     *sigmahat*alpha_opt(:,i))/(1-theta(i)*gamma*alpha_opt(:,i)'*sigmahat*alpha_opt(:,i)) + eta(i)/theta(i),(1:length(eta))');

error=abs(ValofRisk_AN-ValofRisk_EM).^2;  

end 