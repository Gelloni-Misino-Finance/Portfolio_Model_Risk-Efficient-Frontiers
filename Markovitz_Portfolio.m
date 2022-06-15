function [weights,alpha_nominal,ValofRisk_AN,ValofRisk_EM,Expected_return,Volatility]=Markovitz_Portfolio(LogReturns,N_stocks,gamma)
%  
% Function that creates the optimal portfolio, within a pool of assets,
% considering Markovitz model, both using the emprical and analytical
% formulation of the problem.
% 
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


LogReturns=LogReturns(:,1:N_stocks);
N=size(LogReturns,2);
%% Data Preparation

mu=mean(LogReturns,1)';
sigmahat=cov(LogReturns);
A=ones(1,length(mu))*(sigmahat\mu);    %CAMBIARE LE LENGTH UNA VOLTA PROVATA N 
C=ones(1,length(mu))*(sigmahat\ones(length(mu),1));

%% Markovitz Optimization
alpha_nominal= (A/gamma)*(sigmahat\mu)/A + (1-(A/gamma))*(sigmahat\ones(length(mu),1))/C;    

ValofRisk_AN= gamma/2*alpha_nominal'*sigmahat*alpha_nominal-alpha_nominal'*mu;

Z = @(a) (gamma/2)*(a'*sigmahat*a)-a'*mu ;
a0=rand(N,1);
D=[];
b=[];
Deq=ones(1,N);
beq=1;
lb=[] ;
ub= [] ;

% [weights,ValofRisk_EM] = fmincon(Z,a0,D,b,Deq,beq,lb,ub);

rng default % For reproducibility
options=optimoptions("fmincon","MaxFunctionEvaluations",10000,"MaxIterations",3000);
problem = createOptimProblem('fmincon','objective', Z, 'x0', a0, ...
        'Aineq', D, 'bineq', b, 'Aeq', Deq, 'beq', beq, 'lb', lb, ...
        'ub', ub,'options',options);

ms = MultiStart('FunctionTolerance',2e-4,'XTolerance',5e-3,...
    'StartPointsToRun','bounds-ineqs','UseParallel', true);

[weights,ValofRisk_EM] = run(ms,problem,500);

Expected_return = alpha_nominal'*mu;
Volatility = alpha_nominal'*sigmahat*alpha_nominal; %gamma/2

end