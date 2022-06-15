import numpy as np
import scipy as sp
from scipy.optimize import LinearConstraint


class Efficient_Frontier_Portfolios:

  def __init__(self,LogReturns,N_stocks,gamma):
    self.LogReturns = LogReturns
    self.N_stocks = N_stocks
    self.gamma = gamma

  def Markovitz_Portfolio(self,i):
    #
    # Function that creates the optimal portfolio, within a pool of assets,
    # considering Markovitz model, both using the emprical and analytical
    # formulation of the problem.
    #
    #
    # INPUT:

    #   LogReturns:         vector of log returns of all the stocks admissible
    #   Nstocks:            number of stocks selected for the portfolio composition
    #   gamma:              risk adversion coefficient
    #
    # OUTPUT:
    #
    #   weights:            Vector of optimal weights computed numerically
    #   alpha_nominal:      Vector of optimal weights computed analytically
    #   ValofRisk_AN:       Value of Risk computed with the analytical weights
    #   ValofRisk_EM:       Value of Risk computed with the empirical  weights
    #   Expected Return:    Daily expected return
    #   Volatility:         Volatility of our portfolio
    N_stocks= self.N_stocks[i]
    LogReturns = self.LogReturns
    LogReturns = LogReturns[:, :N_stocks]
    gamma=self.gamma
    N =int(np.size(LogReturns,1))

    ## Data Preparation

    mu = np.column_stack(np.mean(LogReturns, axis=0)).T
    sigmahat = np.cov(LogReturns.T)
    A = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigmahat, mu))))
    C = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigmahat, np.ones([N, 1])))))

    weights=np.zeros ([N_stocks,1])
    alpha_nominal=np.zeros ([N_stocks,1])
    # ValofRisk_EM =np.zeros ([N_stocks,1])
    # Expected_return = np.zeros ([N_stocks,1])
    # volatility =np.zeros ([N_stocks,1])
    # ValofRisk_AN= np.zeros ([N_stocks,1])
    ## Markovitz Optimization
    alpha_nominal = (A / gamma) * (np.linalg.solve(sigmahat, mu)) / A + (1 - (A / gamma)) * (np.linalg.solve(sigmahat, np.ones([N, 1]))) / C
    ValofRisk_AN = np.dot(np.dot(np.dot(gamma / 2, alpha_nominal.T), sigmahat), alpha_nominal) - np.dot(alpha_nominal.T,mu)
    alpha_nominal =np.reshape(np.array(alpha_nominal),[N_stocks,1])

    def Z(a):
      return np.dot((gamma / 2), (np.dot(np.dot(a.T, sigmahat), a))) - np.dot(a.T, mu)

    a0 = np.random.rand(N)
    np.random.seed(1)
    lb = 1
    ub = 1
    E = np.ones([1, N])
    Cons = sp.optimize.LinearConstraint(E, lb, ub, keep_feasible=True)
    # noinspection PyTypeChecker
    weights_opt = sp.optimize.minimize(Z, a0, method='trust-constr',constraints=Cons, tol=1e-8)
    weights = np.reshape(np.array(weights_opt.x),[N_stocks,1])
    ValofRisk_EM = weights_opt.fun
    Expected_return = np.dot(alpha_nominal.T, mu)
    volatility = np.dot(np.dot(alpha_nominal.T, sigmahat), alpha_nominal)

    return [weights, alpha_nominal, ValofRisk_AN, ValofRisk_EM, Expected_return, volatility]

  def Worst_Case_Portfolio(self,gamma,eta,i):
    # Function that creates the optimal portfolio, within a pool of assets, considering Worst Case dynamics,
    # both using the empirical and analytical formulation of the problem.
    # # INPUT:
    # % LogReturns: vector of log returns of all the stocks admissible
    # Nstocks: number of stocks selected for the portfolio composition
    # gamma: risk adversion coefficient
    # % OUTPUT:
    # % weights: Vector of optimal weights computed numerically
    # alpha_nominal: Vector of optimal weights computed analitically
    # ValofRisk_AN: Value of Risk computed with the analitical weights
    # ValofRisk_EM: Value of Risk computed with the empirical  weights
    # Expected Return: Daily expected return
    # Volatility: Volatility of our portfolio
    # error: Squared error between the analytical and empirical formulation of the problem
    N_stocks = self.N_stocks[i]
    LogReturns = self.LogReturns[:, :N_stocks]
    N = self.N_stocks[i]
    if self.gamma==gamma:
      gamma=self.gamma

    # Data Preparation

    mu = np.column_stack(np.mean(LogReturns, axis=0)).T
    sigma = np.cov(LogReturns.T)
    A = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigma, mu))))
    C = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigma, np.ones([N, 1])))))
    B = float(np.dot(np.transpose(mu),(np.linalg.solve(sigma, mu))))
    D = float(B * C - A ** 2)

    # Calibration of Theta and S

    np.random.seed(71)

    def Gamma20(x1,x2):
      return (gamma * (1 - gamma * np.dot(x1,x2)) + x1) / (1 - gamma * np.dot(x1,x2) )**2

    def system(x):
        theta=x[0]
        S=x[1]
        return S - 1 / C * (D / Gamma20(theta,S) ** 2 + 1), \
               theta / 2 * S*Gamma20(theta,S) + 1 / 2 * np.log(1 - gamma *theta* S) - eta[i]

    x0 =np.random.random(2)
    theta = np.zeros([np.size(eta)])
    S = np.zeros([np.size(eta)])


    for i in range(0, np.size(eta)):
      x = sp.optimize.least_squares(system, x0=x0, bounds=([0,1/C],[np.inf,1]),method='dogbox',max_nfev=5e3)
      theta[i]=x.x[0]
      S[i]=x.x[1]

    a0 = np.zeros(N)
    np.random.seed(1)
    lb = 1
    ub = 1
    E = np.ones([1, N])
    Cons = sp.optimize.LinearConstraint(E, lb, ub, keep_feasible=True)
    weights = np.zeros([N, np.size(eta)])
    alpha_opt = np.zeros([N, np.size(eta)])
    ValofRisk_EM = np.zeros([np.size(eta)])

    def LagW(a,eta,theta):
        return -(1 / (2 * theta)) * np.log(1 - theta * gamma * (np.dot(np.transpose(a),np.dot(sigma, a)))) - np.dot(np.transpose(a), mu) + 1 / 2 * (theta * (np.dot(np.dot(np.transpose(a), sigma),a))/(1 - theta * gamma * (np.dot(np.dot(np.transpose(a), sigma), a))) + eta / theta)


    for i in range(0,np.size(eta)):

        weights_opt = sp.optimize.minimize(LagW, a0,(eta[i],theta[i]), method='trust-constr', constraints=Cons, tol=1e-8)
        weights[:, i] = weights_opt.x
        ValofRisk_EM[i] = weights_opt.fun

    # Analytical formulation
    ValofRisk_AN = np.zeros([np.size(eta),])
    mu=np.reshape(mu,[N,])
    for i in range(0,np.size(eta)):
      alpha_opt[:,i]    = ((A/ Gamma20(theta[i], S[i])) * (np.linalg.solve(sigma, mu)) )/A + (1 - A / Gamma20(theta[i], S[i]))* (np.linalg.solve(sigma, np.ones(N))) / C
      ValofRisk_AN[i] = -(1 / (2 * theta[i])) * np.log(1 - theta[i] * gamma * (np.dot(np.dot(np.transpose(alpha_opt[:,i]), sigma), alpha_opt[:,i]))) - np.dot(np.transpose(alpha_opt[:,i]), mu) + 0.5 * (theta[i] * (np.dot(np.dot(np.transpose(alpha_opt[:,i]), sigma), alpha_opt[:,i])) / (1 - theta[i] * gamma * (np.dot(np.dot(np.transpose(alpha_opt[:,i]), sigma), alpha_opt[:,i]))) + eta[i] / theta[i])

    error = np.zeros([np.size(eta), ])
    error = abs(ValofRisk_AN-ValofRisk_EM)**2

    return weights,alpha_opt, ValofRisk_AN[:,],ValofRisk_EM[:,],error

  def Worst_Case_PortfolioGX(self, gamma, eta,i):
    #
    # Function that creates the optimal portfolio, within a pool of assets,
    # considering Worst Case dynamics according to Glasserman's model, both using the emprical and analytical
    # formulation of the problem.
    #
    # INPUT:
    #
    #   LogReturns:         vector of log returns of all the stocks admissible
    #   Nstocks:            number of stocks selected for the portfolio composition
    #   gamma:              risk adversion coefficient
    #
    # OUTPUT:
    #
    #   weights:            Vector of optimal weights computed numerically
    #   alpha_nominal:      Vector of optimal weights computed analitically
    #   ValofRisk_AN:       Value of Risk computed with the analitical weights
    #   error:              Squared error between the analytical and empirical formulation of the problem
    #   ValofRisk_EM:       Value of Risk computed with the empirical  weights
    #   theta:              Value of the theta parameter
    N_stocks = self.N_stocks[i]
    LogReturns =self.LogReturns[:, :N_stocks]
    if self.gamma == gamma:
        gamma = self.gamma

    N = np.size(LogReturns, 1)

    ## Data Preparation

    mu = np.column_stack(np.mean(LogReturns, axis=0)).T
    sigma = np.cov(LogReturns.T)
    A = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigma, mu))))
    C = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigma, np.ones([N, 1])))))
    B = float(np.dot(mu.T, (np.linalg.solve(sigma, mu))))
    D = float(B * C - A ** 2)

    ## Calibration of theta and S

    def Gamma20GX(x1,x2):
      return gamma / (1 - gamma * np.dot(x1, x2))

    np.random.seed(71)

    def system(x):
        theta=x[0]
        S=x[1]
        return S - 1 / C * (D / Gamma20GX(theta,S) ** 2 + 1),\
               0.5 * (Gamma20GX(theta,S) / gamma - 1 - np.log(Gamma20GX(theta,S) / gamma)) - eta[i]

    x0 = np.random.random(2)
    theta = np.zeros([np.size(eta),])
    S = np.zeros([np.size(eta),])

    for i in range(0, np.size(eta)):
        x = sp.optimize.least_squares(system, x0=x0, bounds=([0, 1 / C], [np.inf, 1]),method='dogbox',max_nfev=5e3,gtol=1e-8)
        theta[i] = x.x[0]
        S[i] = x.x[1]

    ## Empirical Formulation of the problem

    def Lag(a,eta,theta):
        return -(1 / (2 * theta)) * np.log(1 - theta * gamma * np.dot(np.transpose(a), np.dot(sigma, a))) - np.dot(np.transpose(a),mu) + eta / theta

    weights_GX = np.zeros([N, np.size(eta)])
    ValofRisk_EM = np.zeros([np.size(eta),])
    ValofRisk_AN = np.zeros([np.size(eta),],dtype=float)
    alpha_opt = np.zeros([N, np.size(eta)])
    a0 = (1/N)*np.ones([N,])
    lb = 1
    ub = 1
    E = np.ones([1, N])
    for i in range(np.size(eta)):
        Cons = sp.optimize.LinearConstraint(E, lb, ub, keep_feasible=True)
        weights_EM_GX = sp.optimize.minimize(Lag, a0,(eta[i],theta[i]), method='SLSQP', constraints=Cons, tol=1e-8,)
        weights_GX[:,i] = weights_EM_GX.x
        ValofRisk_EM[i] = weights_EM_GX.fun
    mu=np.reshape(mu,[N,])
    # Analytical formulation
    for i in range(0,np.size(eta)):
      alpha_opt[:,i] = A / np.dot(Gamma20GX(theta[i], S[i]), (np.linalg.solve(sigma, mu))) / A + np.dot((1 - A / Gamma20GX(theta[i], S[i])), (np.linalg.solve(sigma, np.ones(N)))) / C
      ValofRisk_AN[i] = (1 / (2 * C) * (Gamma20GX(theta[i], S[i]) - D / Gamma20GX(theta[i], S[i])) - A / C)

    error = abs(ValofRisk_AN - ValofRisk_EM)** 2

    return[weights_GX,alpha_opt,ValofRisk_AN,ValofRisk_EM,error,theta]  ##,theta,S

  def Best_Case_Portfolio(self, gamma, eta,i):

      # Function that creates the optimal portfolio, within a pool of assets, considering Best Case dynamics,
      # both using the empirical and analytical formulation of the problem.
      # # INPUT:
      # % LogReturns: vector of log returns of all the stocks admissible
      # Nstocks: number of stocks selected for the portfolio composition
      # gamma: risk adversion coefficient
      # eta: ray of the KL ball

      # OUTPUT:
      # weights: Vector of optimal weights computed numerically
      # ValofRisk_EM: Value of Risk computed with the empirical  weights

      N_stocks = self.N_stocks[i]
      LogReturns = self.LogReturns[:, :N_stocks]
      if self.gamma == gamma:
          gamma = self.gamma

      N = N_stocks

      ## Data Preparation

      mu = np.column_stack(np.mean(LogReturns, axis=0)).T
      sigma = np.cov(LogReturns.T)
      A = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigma, mu))))
      C = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigma, np.ones([N, 1])))))
      B = float(np.dot(np.transpose(mu), (np.linalg.solve(sigma, mu))))
      D = float(B * C - A ** 2)

      ## Calibration of Theta and S

      np.random.seed(71)

      def Gamma20_B(x1,x2):

        return (gamma * (1 - gamma * x1 * x2) + x1) / (1 - gamma * x1 * x2) ** 2



      def system_B(x):
          return 0.5 * np.dot(np.dot(x[0],x[1] ), Gamma20_B(x[0],x[1]))+ 0.5 * np.log(1 - gamma * np.dot(x[0],x[1] )) - eta[i], x[1] - 1 / C * (D / Gamma20_B(x[0],x[1]) ** 2 + 1)

      x0 = np.reshape([-np.random.random(1), np.random.random(1)], 2)
      theta = np.zeros([np.size(eta)])
      S = np.zeros([np.size(eta)])

      for i in range(0, np.size(eta)):
          x = sp.optimize.least_squares(system_B, x0=x0, bounds=([-np.inf, 1 / C], [0, 1]),method='dogbox',max_nfev=5e3)
          theta[i] = x.x[0]
          S[i] = x.x[1]

      ## Empirical Implementation

      def sigmatilda(a, theta):
        return np.linalg.inv(np.linalg.inv(sigma) - gamma * np.dot(theta, np.dot(a, np.transpose(a))))

      def Q(a,theta):#
        return np.dot(1 / np.sqrt(1-theta*gamma*np.dot(np.transpose(a),np.dot(sigma,a)) ), np.exp(- theta * np.dot(np.transpose(a), mu) + 0.5 * (theta ** 2) * np.dot(np.dot(np.transpose(a), sigma), a)/(1-theta*gamma*np.dot(np.transpose(a),np.dot(sigma,a)))))

      a0 = np.random.rand(int(N))
      np.random.seed(71)
      lb = 1
      ub = 1
      E = np.ones([1, N])
      Cons = sp.optimize.LinearConstraint(E, lb, ub, keep_feasible=True)
      weights = np.zeros([N_stocks, np.size(eta)])
      ValofRisk_EM_B = np.zeros([np.size(theta), 1])

      def LagB(a,theta,eta):#
        return np.dot(1 / theta, np.log(Q(a, theta))) + eta / theta

      for i in range(np.size(eta)):
        weights_opt = sp.optimize.minimize(LagB, a0,(theta[i],eta[i]), method='SLSQP', constraints=Cons, tol=1e-8)
        weights[:, i] = weights_opt.x
        ValofRisk_EM_B[i] = weights_opt.fun

      return weights, ValofRisk_EM_B

  def Best_Case_PortfolioGX(self, gamma, eta,i):
      #
      # Function that creates the optimal portfolio, within a pool of assets,
      # considering Worst Case dynamics according to Glasserman's model, both using the emprical and analytical
      # formulation of the problem.
      #
      # INPUT:
      #
      #   LogReturns:         vector of log returns of all the stocks admissible
      #   Nstocks:            number of stocks selected for the portfolio composition
      #   gamma:              risk adversion coefficient
      #
      # OUTPUT:
      #
      #   weights:            Vector of optimal weights computed numerically
      #   alpha_nominal:      Vector of optimal weights computed analitically
      #   ValofRisk_AN:       Value of Risk computed with the analitical weights
      #   error:              Squared error between the analytical and empirical formulation of the problem
      #   ValofRisk_EM:       Value of Risk computed with the empirical  weights
      #   theta:              Value of the theta parameter
      N_stocks = self.N_stocks[i]
      LogReturns = self.LogReturns[:, :N_stocks]
      if self.gamma == gamma:
          gamma = self.gamma

      N = np.size(LogReturns, 1)

      ## Data Preparation

      mu = np.column_stack(np.mean(LogReturns, axis=0)).T
      sigma = np.cov(LogReturns.T)
      A = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigma, mu))))
      C = float(np.dot(np.ones([1, N]), (np.linalg.solve(sigma, np.ones([N, 1])))))
      B = float(np.dot(mu.T, (np.linalg.solve(sigma, mu))))
      D = float(B * C - A ** 2)

      ## Calibration of theta and S

      def Gamma20_BGX(x1, x2):
          return gamma / (1 - gamma * np.dot(x1, x2))

      np.random.seed(71)

      def GammaK(k):
         return gamma /(1-k*gamma)

      def system_BGX1(x):
            return 0.5*np.dot(x,GammaK(x)) + 0.5* np.log(1-x*gamma) - eta[i]

      x0 = -np.random.random(1)
      k=np.zeros(np.size(eta))
      for i in range(0, np.size(eta)):
          K = sp.optimize.least_squares(system_BGX1, x0, bounds=([-np.inf], [0]),method='dogbox',max_nfev=5e3)
          k[i] = K.x[0]

      x0 = np.reshape([-np.random.random(1),np.random.random(1)],[2])

      def system_BGX2(x):
        return k[i]-np.dot(x[0],x[1]), 0.5 * np.dot(np.dot(x[0],x[1]), GammaK(np.dot(x[0],x[1]))) + 0.5 * np.log(1 - np.dot(x[0],x[1]) * gamma) - eta[i]

      theta = np.zeros([np.size(eta)])
      S = np.zeros([np.size(eta)])
      LB=-np.inf
      for i in range(0, np.size(eta)):
          params = sp.optimize.least_squares(system_BGX2, x0=x0, bounds=([LB,1/C], [0,1]),method='dogbox',max_nfev=5e3)
          theta[i] = params.x[0]
          S[i] = params.x[1]

      def LagB_GX(a, eta, theta):
          return -(1 / (2 * theta)) * np.log(1 - theta * gamma * np.dot(np.transpose(a), np.dot(sigma, a))) - np.dot(
              np.transpose(a), mu) + eta / theta

      weights_BGX = np.zeros([N, np.size(eta)])
      ValofRisk_EM_BGX = np.zeros([np.size(eta), 1])
      a0 = (1 / N) * np.ones([N,])
      lb = 1
      ub = 1
      E = np.ones([1, N])

      for i in range(np.size(eta)):
          Cons = sp.optimize.LinearConstraint(E, lb, ub, keep_feasible=True)
          weights_EM_BGX = sp.optimize.minimize(LagB_GX, a0, (eta[i], theta[i]), method='SLSQP', constraints=Cons,tol=1e-7)
          weights_BGX[:, i] = weights_EM_BGX.x
          ValofRisk_EM_BGX[i, 0] = weights_EM_BGX.fun

      return weights_BGX, ValofRisk_EM_BGX