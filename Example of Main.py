import pandas as pd
import scipy as sp
import scipy.stats as stat
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.optimize import LinearConstraint
import Efficient_Frontier_Portfolios as ptf


# Main Final Project Financial Engineering
# AY2021-2022
# Gelloni Gregorio,Misino Bianca
format('long')
## Data Setting
formatDate='dd/MM/yyyy'
Quotes=pd.read_excel('EUROSTOCK_FINAL.xlsx',sheet_name='Foglio2',decimal=',')
Capitalization=pd.read_excel('Capitalization.xlsx',sheet_name='Foglio2',decimal=',')
date=pd.to_datetime('04/01/2010',dayfirst=True)
Capitalization=Capitalization.sort_values('Capitalization',ascending=False,axis=0)
index=np.array(Capitalization.index)
Quotes=np.array(Quotes)
idx =int(np.float64(np.where(Quotes[:,0]==date)))
Quotes =np.array(Quotes[idx:,1:])
Quotes = Quotes[:,index]
ratio= np.float64(Quotes[1:,:]/Quotes[:-1,:])
LogReturns=np.log(ratio)
# CS1
N_stocks = np.array([10,20,30,46])
gamma = 1
ValofRisk_AN_1=np.empty([4,1], dtype=float, order='F')
ValofRisk_EM_1=np.empty([4,1], dtype=float, order='F')
Expected_Return=np.empty([4,1], dtype=float, order='F')
Volatility=np.empty([4,1], dtype=float, order='F')
weights_1=np.empty([46,4], dtype=float, order='F')
alpha_opt_1=np.empty([46,4], dtype=float, order='F')
eta_max=0.3
ptf1=ptf.Efficient_Frontier_Portfolios(LogReturns,N_stocks,gamma)
for i in range(1,np.size(N_stocks)+1):
 weights_1[:N_stocks[i-1],[i-1]],alpha_opt_1[:N_stocks[i-1],[i-1]],ValofRisk_AN_1[[i-1],0],ValofRisk_EM_1[[i-1],0],Expected_Return[[i-1],0],Volatility[[i-1],0]=ptf.Efficient_Frontier_Portfolios.Markovitz_Portfolio(ptf1,i-1)

## CS2_BEST and WORST GENERAL CASE
    
eta=np.arange(10**-10, 0.31, 0.01)
ValofRisk_AN_W1_10=np.zeros([np.size(eta),])
ValofRisk_AN_W1_20=np.zeros([np.size(eta),])
ValofRisk_AN_W1_30=np.zeros([np.size(eta),])
ValofRisk_AN_W1_46=np.zeros([np.size(eta),])

ValofRisk_EM_W1_10=np.zeros([np.size(eta),])
ValofRisk_EM_W1_20=np.zeros([np.size(eta),])
ValofRisk_EM_W1_30=np.zeros([np.size(eta),])
ValofRisk_EM_W1_46=np.zeros([np.size(eta),])

ValofRisk_EM_B1_10=np.zeros([np.size(eta),])
ValofRisk_EM_B1_20=np.zeros([np.size(eta),])
ValofRisk_EM_B1_30=np.zeros([np.size(eta),])
ValofRisk_EM_B1_46=np.zeros([np.size(eta),])
errorW1 = np.zeros([np.size(eta),4])
weights_W1_10 = np.zeros([N_stocks[0],np.size(eta)],dtype=float)
weights_W1_20 = np.zeros([N_stocks[1],np.size(eta)],dtype=float)
weights_W1_30 = np.zeros([N_stocks[2],np.size(eta)],dtype=float)
weights_W1_46 = np.zeros([N_stocks[3],np.size(eta)],dtype=float)
alpha_opt_W1_10=np.zeros([N_stocks[0],np.size(eta)],dtype=float)
alpha_opt_W1_20=np.zeros([N_stocks[1],np.size(eta)],dtype=float)
alpha_opt_W1_30=np.zeros([N_stocks[2],np.size(eta)],dtype=float)
alpha_opt_W1_46=np.zeros([N_stocks[3],np.size(eta)],dtype=float)
weights_B1_10=np.zeros([N_stocks[0],np.size(eta)],dtype=float)
weights_B1_20=np.zeros([N_stocks[1],np.size(eta)],dtype=float)
weights_B1_30=np.zeros([N_stocks[2],np.size(eta)],dtype=float)
weights_B1_46=np.zeros([N_stocks[3],np.size(eta)],dtype=float)

for i in range(0,np.size(N_stocks)+1):
  if i==0:
       weights_W1_10,alpha_opt_W1_10,ValofRisk_AN_W1_10,ValofRisk_EM_W1_10,errorW1[:,0]=ptf.Efficient_Frontier_Portfolios.Worst_Case_Portfolio(ptf1,gamma,eta,i)
       weights_B1_10,ValofRisk_EM_B1_10=ptf.Efficient_Frontier_Portfolios.Best_Case_Portfolio(ptf1,gamma,eta,i)
  elif i==1:
       weights_W1_20,alpha_opt_W1_20,ValofRisk_AN_W1_20,ValofRisk_EM_W1_20,errorW1[:,1]= ptf.Efficient_Frontier_Portfolios.Worst_Case_Portfolio(ptf1,gamma,eta,i)
       weights_B1_20,ValofRisk_EM_B1_20=ptf.Efficient_Frontier_Portfolios.Best_Case_Portfolio(ptf1,gamma,eta,i)
  elif i==2:
       weights_W1_30,alpha_opt_W1_30,ValofRisk_AN_W1_30,ValofRisk_EM_W1_30,errorW1[:,2]= ptf.Efficient_Frontier_Portfolios.Worst_Case_Portfolio(ptf1,gamma,eta,i)
       weights_B1_30,ValofRisk_EM_B1_30=ptf.Efficient_Frontier_Portfolios.Best_Case_Portfolio(ptf1,gamma,eta,i)
  elif i==3:
       weights_W1_46,alpha_opt_W1_46,ValofRisk_AN_W1_46,ValofRisk_EM_W1_46,errorW1[:,3]= ptf.Efficient_Frontier_Portfolios.Worst_Case_Portfolio(ptf1,gamma,eta,i)
       weights_B1_46,ValofRisk_EM_B1_46=ptf.Efficient_Frontier_Portfolios.Best_Case_Portfolio(ptf1,gamma,eta,i)

## SECTION GX
ValofRisk_AN_W1GX_10=np.zeros([np.size(eta),])
ValofRisk_AN_W1GX_20=np.zeros([np.size(eta),])
ValofRisk_AN_W1GX_30=np.zeros([np.size(eta),])
ValofRisk_AN_W1GX_46=np.zeros([np.size(eta),])

ValofRisk_EM_W1GX_10=np.zeros([np.size(eta),])
ValofRisk_EM_W1GX_20=np.zeros([np.size(eta),])
ValofRisk_EM_W1GX_30=np.zeros([np.size(eta),])
ValofRisk_EM_W1GX_46=np.zeros([np.size(eta),])

ValofRisk_EM_B1GX_10=np.zeros([np.size(eta),])
ValofRisk_EM_B1GX_20=np.zeros([np.size(eta),])
ValofRisk_EM_B1GX_30=np.zeros([np.size(eta),])
ValofRisk_EM_B1GX_46=np.zeros([np.size(eta),])
errorW1GX = np.zeros([np.size(eta),4])
weights_W1GX_10 = np.zeros([N_stocks[0],np.size(eta)],dtype=float)
weights_W1GX_20 = np.zeros([N_stocks[1],np.size(eta)],dtype=float)
weights_W1GX_30 = np.zeros([N_stocks[2],np.size(eta)],dtype=float)
weights_W1GX_46 = np.zeros([N_stocks[3],np.size(eta)],dtype=float)
alpha_opt_W1GX_10=np.zeros([N_stocks[0],np.size(eta)],dtype=float)
alpha_opt_W1GX_20=np.zeros([N_stocks[1],np.size(eta)],dtype=float)
alpha_opt_W1GX_30=np.zeros([N_stocks[2],np.size(eta)],dtype=float)
alpha_opt_W1GX_46=np.zeros([N_stocks[3],np.size(eta)],dtype=float)
weights_B1GX_10=np.zeros([N_stocks[0],np.size(eta)],dtype=float)
weights_B1GX_20=np.zeros([N_stocks[1],np.size(eta)],dtype=float)
weights_B1GX_30=np.zeros([N_stocks[2],np.size(eta)],dtype=float)
weights_B1GX_46=np.zeros([N_stocks[3],np.size(eta)],dtype=float)

for i in range(0,np.size(N_stocks)+1):
  if i==0:
       weights_W1GX_10,alpha_opt_W1GX_10,ValofRisk_AN_W1GX_10,ValofRisk_EM_W1GX_10,errorW1GX[:,0],t=ptf.Efficient_Frontier_Portfolios.Worst_Case_PortfolioGX(ptf1,gamma,eta,i)
       weights_B1GX_10,ValofRisk_EM_B1GX_10=ptf.Efficient_Frontier_Portfolios.Best_Case_PortfolioGX(ptf1,gamma,eta,i)
  elif i==1:
       weights_W1GX_20,alpha_opt_W1GX_20,ValofRisk_AN_W1GX_20,ValofRisk_EM_W1GX_20,errorW1GX[:,1],t= ptf.Efficient_Frontier_Portfolios.Worst_Case_PortfolioGX(ptf1,gamma,eta,i)
       weights_B1GX_20,ValofRisk_EM_B1GX_20=ptf.Efficient_Frontier_Portfolios.Best_Case_PortfolioGX(ptf1,gamma,eta,i)
  elif i==2:
       weights_W1GX_30,alpha_opt_W1GX_30,ValofRisk_AN_W1GX_30,ValofRisk_EM_W1GX_30,errorW1GX[:,2],t= ptf.Efficient_Frontier_Portfolios.Worst_Case_PortfolioGX(ptf1,gamma,eta,i)
       weights_B1GX_30,ValofRisk_EM_B1GX_30=ptf.Efficient_Frontier_Portfolios.Best_Case_PortfolioGX(ptf1,gamma,eta,i)
  elif i==3:
       weights_W1GX_46,alpha_opt_W1GX_46,ValofRisk_AN_W1GX_46,ValofRisk_EM_W1GX_46,errorW1GX[:,3],t= ptf.Efficient_Frontier_Portfolios.Worst_Case_PortfolioGX(ptf1,gamma,eta,i)
       weights_B1GX_46,ValofRisk_EM_B1GX_46=ptf.Efficient_Frontier_Portfolios.Best_Case_PortfolioGX(ptf1,gamma,eta,i)



## CS3

format('long')
## Data Setting
formatDate='dd/MM/yyyy'
Dividend=pd.read_excel('EUROSTOVK_DIVIDENDS_FINAL.xlsx',sheet_name='Tabella1_3',decimal=',')
Capitalization=pd.read_excel('Capitalization.xlsx',sheet_name='Foglio2',decimal=',')
date=pd.to_datetime('04/01/2010',dayfirst=True)
Capitalization=Capitalization.sort_values('Capitalization',ascending=False,axis=0)
index=np.array(Capitalization.index)
Dividend=np.array(Dividend)
idx =int(np.float64(np.where(Dividend[:,0]==date)))
Dividend =np.array(Dividend[idx:,1:])
Dividend = Dividend[:,index]
Quotes_DV= Quotes + Dividend

ratio= np.float64(Quotes_DV[1:,:]/Quotes_DV[:-1,:])
LogReturns_DV=np.log(ratio)

#CS3

ValofRisk_AN_DV=np.empty([4,1], dtype=float, order='F')
ValofRisk_EM_DV=np.empty([4,1], dtype=float, order='F')
Expected_Return_DV=np.empty([4,1], dtype=float, order='F')
Volatility_DV=np.empty([4,1], dtype=float, order='F')
weights_DV=np.empty([46,4], dtype=float, order='F')
alpha_opt_DV=np.empty([46,4], dtype=float, order='F')
eta_max=0.3


ptf_DV=ptf.Efficient_Frontier_Portfolios(LogReturns_DV,N_stocks,gamma)
for i in range(1,np.size(N_stocks)+1):
 weights_DV[:N_stocks[i-1],[i-1]],alpha_opt_DV[:N_stocks[i-1],[i-1]],ValofRisk_AN_DV[[i-1],0],ValofRisk_EM_DV[[i-1],0],Expected_Return_DV[[i-1],0],Volatility_DV[[i-1],0]=ptf.Efficient_Frontier_Portfolios.Markovitz_Portfolio(ptf_DV,i-1)


N_stocks = np.array([10,20,30,46])
gamma = 1


## Q0

mu = np.column_stack(np.mean(LogReturns, axis=0)).T
sigma = np.cov(LogReturns.T)

A = float(np.dot(np.ones([1, N_stocks[3]]), (np.linalg.solve(sigma, mu))))
C = float(np.dot(np.ones([1, N_stocks[3]]), (np.linalg.solve(sigma, np.ones([N_stocks[3], 1])))))
B = float(np.dot(np.transpose(mu),(np.linalg.solve(sigma, mu))))
D = float(B * C - A ** 2)


def s_Q(m):
    return 1 / C + C / D * (m - A / C)**2




def normfit(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    sigma = np.std(data, ddof=1)
    return m, sigma


MU,CI_MU= normfit(np.dot(np.transpose(alpha_opt_DV[:,3]),np.transpose(LogReturns_DV)))
m_Q0=np.linspace(MU-CI_MU,MU+CI_MU,200)

fig = plt.figure()
plot(s_Q(m_Q0),m_Q0, label="Efficient Frontier")
plt.scatter(Volatility_DV[3,0],Expected_Return_DV[3,0])
xlabel('Var')
ylabel('E[x]')
title('Efficient Frontier Nominal Case')
plt.show()


#Q1
#weights_DV=np.zeros([np.size(eta),])
#alpha_opt_DV=np.zeros([np.size(eta),])
#ValofRisk_AN_DV=np.zeros([np.size(eta),])
#ValofRisk_EM_DV=np.zeros([np.size(eta),])
#errorDV=np.zeros([np.size(eta),])
#thetaDV=np.zeros([np.size(eta),])
weights_DV, alpha_opt_DV, ValofRisk_AN_DV, ValofRisk_EM_DV, errorDV,thetaDV = ptf.Efficient_Frontier_Portfolios.Worst_Case_PortfolioGX(ptf_DV, gamma, eta,3)
MU=np.zeros([np.size(eta),])
CI_MU=np.zeros([np.size(eta),])
m_Q1=np.zeros([np.size(eta),200])
for i in range(0,np.size(eta)):
    MU[i],CI_MU[i]= normfit(np.dot(np.transpose(alpha_opt_DV[:,i]),np.transpose(LogReturns_DV)))
    m_Q1[i,:]=np.linspace(MU[i]-CI_MU[i],MU[i]+CI_MU[i],200)
thetaDV=np.reshape(thetaDV,[np.size(eta),1])*np.ones([np.size(eta),200])
fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(1,1,1, projection='3d')
p = ax.plot_surface(s_Q(m_Q1),thetaDV , m_Q1, linewidth=0.2, rstride=4, cstride=4, alpha=0.75)
plt.xlabel('Var')
plt.ylabel('theta')
#plt.zlabel('E[x]')
title('Efficient Frontier for different values of theta')
ax.view_init(71, 71)
plt.show()


#Q2
mu=reshape(mu,[N_stocks[3],])
eta = [0.05,0.10]
def Gamma(theta, gamma, S):
    return gamma / (1 - np.dot(np.dot(theta,gamma), S))
def Gamma39(theta, gamma):
    return (gamma * C + np.sqrt((gamma * C) ** 2 + 4 * np.dot(np.dot(gamma,theta),(C - np.dot(theta, gamma))) * D)) / (2 * (C - np.dot(theta, gamma)))
def alpha_opt(theta, gamma, S):
    return (A / Gamma(theta, gamma, S))*(np.linalg.solve(sigma, mu)) / A + (1 - A / Gamma(theta, gamma, S))* (np.linalg.solve(sigma, np.ones(N_stocks[3],)))/C

def R(theta, gamma, S):
    return 0.5 * (Gamma(theta, gamma, S) / gamma - 1 - np.log(Gamma(theta, gamma, S) / gamma))
np.random.seed(71)

def R2(k,eta):
 return 0.5 * (1 / k - 1 - log(1 / k))- eta

k = np.zeros(np.size(eta),)
x0=2
for i in range(0,np.size(eta)):
    K=sp.optimize.least_squares(R2, x0,args=[eta[i]])
    k[i]=K.x[0]



x0 = np.random.rand(4,)
z = np.zeros([2, 4])


def sysQ2(z,k,eta):
    return (z[0]* np.dot(sigma ,alpha_opt(z[2], z[0], z[3]))[0] - mu[0] + z[1]* np.ones([N_stocks[3],])[0],
            z[0] * np.dot(np.transpose(alpha_opt(z[2], z[0], z[3])), np.dot(sigma,alpha_opt(z[2],z[0],z[3])) ) - np.dot(np.transpose(mu), alpha_opt(z[2], z[0], z[3]))+ z[1],
            z[3] - (1 / C) * (D / Gamma39(z[2], z[0]) ** 2 + 1),
            k - 1 + np.dot(np.dot(z[0],z[2]), z[3]),
            0.5 * (Gamma39(z[2], z[0]) / z[0] - 1 - np.log(Gamma39(z[2], z[0]) / z[0])) - eta)


ub=np.array([np.inf, np.inf, np.inf, 1])
lb=np.array([0, 0, 0, 1/C])

for i in range(0, np.size(eta)):
    Z = sp.optimize.least_squares(sysQ2,x0, method='trf', bounds=(lb,ub),args=(k[i],eta[i]))
    z[i,:] = Z.x[:,]

gamma  = z[:,0]
#lam = z[:, 1]
theta = z[:,2]
S = z[:,3]


def s_Q2(m,k):
    return (1 / C + C / D * (m - A / C)**2) / k

alpha_opt_Q2=np.zeros([N_stocks[3],np.size(eta)])

for i in range(0,np.size(eta)):
    alpha_opt_Q2[:,i] = alpha_opt(theta[i], gamma[i], S[i])

MU_Q2=np.zeros(np.size(eta),)
CI_MU_Q2=np.zeros(np.size(eta),)

for i in range(0,1):
    MU_Q2[i],CI_MU_Q2[i] = normfit(np.dot(np.transpose(alpha_opt_Q2[:,i]), np.transpose(LogReturns_DV)))



fig4 = plt.figure()
plot(s_Q2(m_Q0,k[0]),m_Q0, label="Efficient Frontier")
plt.scatter(S[0],MU[0])
xlabel('Var')
ylabel('E[x]')
title('Efficient Frontier Nominal Case for eta=5%')
plt.show()

fig5 = plt.figure()
plot(s_Q2(m_Q0,k[1]),m_Q0, label="Efficient Frontier")
plt.scatter(S[1],MU[1])
xlabel('Var')
ylabel('E[x]')
title('Efficient Frontier Nominal Case for eta=10%')
plt.show()


#Q3
k = np.zeros(np.size(eta),)
x0=2
for i in range(0,np.size(eta)):
    K=sp.optimize.least_squares(R2, x0,args=[eta[i]])
    k[i]=K.x[0]


def sysQ2(z,k,eta):
    return (np.dot(z[0], np.dot(sigma ,alpha_opt(z[2], z[0], z[3])) )[0] - mu[0] + np.dot(z[1], ones([N_stocks[3],]))[0] ,
            z[0] * np.dot(np.transpose(alpha_opt(z[2], z[0], z[3])), np.dot(sigma,alpha_opt(z[2],z[0],z[3])) )- np.dot(np.transpose(mu) , alpha_opt(z[2], z[0], z[3]))+ z[1] ,
            z[3] - (1 / C) * (D / Gamma39(z[2], z[0]) ** 2 + 1),
            k - 1 + np.dot(np.dot(z[0],z[2]), z[3]),
            0.5 * (Gamma39(z[2], z[0]) / z[0] - 1 - np.log(Gamma39(z[2], z[0]) / z[0])) - eta)

z = np.zeros([2, 4])
x0 =[1,1,-1,0.0042]
ub=np.array([np.inf, np.inf, 0, 1])
lb=np.array([0, 0, -np.inf, 1/C])

#Cons = sp.optimize.LinearConstraint(E, lb, ub, keep_feasible=True)

for i in range(0, np.size(eta)):
    Z = sp.optimize.least_squares(sysQ2,x0,method='trf', bounds=(lb,ub),args=(k[i],eta[i]))
    z[i,:] = Z.x[:,]

gamma  = z[:,0]
#lam = z[:, 1]
theta = z[:,2]
S = z[:,3]


def s_Q3 (m, k):
    return (1 / C + C / D * (m - A / C)** 2)/ k

weights_BGX=np.zeros([N_stocks[3],np.size(eta)])
ValofRisk_EM_BGX=np.zeros(np.size(eta),)
MU_Q3=np.zeros(N_stocks[3],)
S_Q3=np.zeros(N_stocks[3],)
for i in range(0,np.size(eta)):
    gamma_GX=gamma[i]
    weights_BGX1, ValofRisk_EM_BGX[i] = ptf.Efficient_Frontier_Portfolios.Best_Case_PortfolioGX(ptf_DV, gamma_GX, [eta[i]],3)
    weights_BGX[:, i]=np.reshape(weights_BGX1,[46,])
    MU_Q3[i] = np.dot(np.transpose(weights_BGX[:, i]),mu)
    S_Q3[i] =np.dot(np.dot(np.transpose(weights_BGX[:, i]),sigma),weights_BGX[:,i])

fig6 = plt.figure()
plot(s_Q3(m_Q0,k[0]),m_Q0, label="Efficient Frontier")
plt.scatter(S[0],MU[0])
xlabel('Var')
ylabel('E[x]')
title('Efficient Frontier Nominal Case for eta=5%')
plt.show()

fig7 = plt.figure()
plot(s_Q3(m_Q0,k[1]),m_Q0, label="Efficient Frontier")
plt.scatter(S[1],MU[1])
xlabel('Var')
ylabel('E[x]')
title('Efficient Frontier Nominal Case for eta=10%')
plt.show()

#Q4 What is the shape of the frontier of m as a function of s?

def GammaW(theta, gamma, S):
    return (gamma * (1 - gamma * np.dot(theta, S)) + theta) / (1 - gamma * np.dot(theta, S)) ** 2

eta=np.arange(10**-10, 0.31, 0.01)

#GammaW     = @(theta,gamma,S) (gamma*(1-theta*gamma*S) + theta)/(1-theta*gamma*S);
#alpha_opt = @(theta,gamma,S) A/GammaW(theta,gamma,S) *(sigma\mu)/A+(1-A/GammaW(theta,gamma,S))*(sigma\ones(N_stocks,1))/C;     %(28)
def alpha_opt(theta,gamma,S):
    return((A / GammaW(theta,gamma, S)) * (np.linalg.solve(sigma, mu))) / A + (
            1 - A / GammaW(theta,gamma, S)) * (np.linalg.solve(sigma, np.ones(N_stocks[3]))) / C

#s_Q       = @(m) 1/C+C/D*(m-A/C).^2; corretta

# Find the optimal gamma
# z(1)= GAMMA
# z(2)= COEFFICIENTE LAGRANGIANO
# z(3)= THETA
# z(4)= S

lb=[0,0,0,1/C]
ub=[np.inf,np.inf,np.inf,1]
x0=np.random.rand(4,)

def sysQ4(z,eta):
    return (z[0]* np.dot(sigma ,alpha_opt(z[2], z[0], z[3]))[0] - mu[0] + z[1]* ones([N_stocks[3],])[0],
            z[0] * np.dot(np.transpose(alpha_opt(z[2], z[0], z[3])), np.dot(sigma,alpha_opt(z[2],z[0],z[3])) )- np.dot(np.transpose(mu), alpha_opt(z[2], z[0], z[3]))+ z[1],
            z[3] - (1 / C) * (D / Gamma39(z[2], z[0]) ** 2 + 1),
            (np.dot(np.transpose(alpha_opt(z[2],z[0],z[3])),np.ones(N_stocks[3],)) - 1),
            z[2]/2*z[3]*GammaW(z[2],z[0],z[3]) + 0.5*np.log(1-z[2]*z[0]*z[3]) - eta )

gamma=np.zeros([np.size(eta),])
lam  =np.zeros([np.size(eta),])
theta=np.zeros([np.size(eta),])
S    =np.zeros([np.size(eta),])
alpha_opt_Q4=np.zeros([N_stocks[3],np.size(eta)])
MU=np.zeros([np.size(eta),])
CI_MU=np.zeros([np.size(eta),])
m_Q4=np.zeros([np.size(eta),200])
MU_Q4=np.zeros([np.size(eta),])
S_Q4=np.zeros([np.size(eta),])

for i in range(0,np.size(eta)):
    O = sp.optimize.least_squares( sysQ4,x0,bounds=(lb,ub),args=[eta[i]])
    gamma[i]    = O.x[0]
    lam[i]      = O.x[1]
    theta[i]    = O.x[2]
    S[i]        = O.x[3]
    alpha_opt_Q4[:,i] = alpha_opt(theta[i],gamma[i],S[i])
    [MU[i],CI_MU[i]]  = normfit((np.dot(np.transpose(alpha_opt_Q4[:,i]),np.transpose(LogReturns_DV))))
    m_Q4[i,:]         = np.linspace(MU[i]-CI_MU[i],MU[i]+CI_MU[i],200)
    MU_Q4[i]          = np.dot(np.transpose( alpha_opt_Q4[:,i]),mu)
    S_Q4[i]           = np.dot(np.dot(np.transpose(alpha_opt_Q4[:,i]),sigma),alpha_opt_Q4[:,i])



theta=np.reshape(theta,[np.size(eta),1])*np.ones([np.size(eta),200])

fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(1,1,1, projection='3d')
p = ax.plot_surface(s_Q(m_Q1),theta,m_Q4,  linewidth=0.2, rstride=4, cstride=4, alpha=0.75)
ax.scatter(S_Q4,theta[:,0],MU_Q4)
plt.xlabel('Var')
plt.ylabel('theta')
#plt.zlabel('E[x]')
title('Efficient Frontier for different values of theta')
ax.view_init(69,120)
plt.show()


#theta=np.reshape(theta,[np.size(eta),1])*np.ones([np.size(eta),200])
fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(1,1,1, projection='3d')
p1 = ax.plot_surface(s_Q(m_Q4),thetaDV , m_Q4, linewidth=0.2, rstride=4, cstride=4, alpha=0.75)
p2=plt.scatter(S_Q4,MU_Q4)
plt.xlabel('Var')
plt.ylabel('theta')
#plt.zlabel('E[x]')
title('Efficient Frontier for different values of theta')
ax.view_init(71, 71)
plt.show()

#Q5


eta = [0.05, 0.10]

def s_Q5(m,S,theta,gamma):
    return (1 / C + C / D * (m + (1 / gamma) * theta * S * GammaW(theta, gamma, S) - A / C) ** 2) / S


lb = np.array([0, 0, 0, 1 / C])
ub = np.array([inf, inf, inf, 1])
x0 = np.random.rand(4,)
z= np.zeros(4,)
gamma=np.zeros([np.size(eta),])
lam  =np.zeros([np.size(eta),])
theta=np.zeros([np.size(eta),])
S    =np.zeros([np.size(eta),])
for i in range(0,np.size(eta)):
    O = sp.optimize.least_squares( sysQ4,x0,bounds=(lb,ub),args=[eta[i]])
    gamma[i]    = O.x[0]
    lam[i]      = O.x[1]
    theta[i]    = O.x[2]
    S[i]        = O.x[3]

def sigmatilda(a, theta, gamma):
    return np.linalg.inv(np.linalg.inv(sigma) - theta * gamma * np.dot(a , np.transpose(a)))

def mutilda(a, theta, gamma):
    return mu - theta * np.dot(sigmatilda(a, theta, gamma) ,a)

def function(stilda,mtilda,theta):
    return 1 / C + C / D * (mtilda**2 * (theta * stilda)**2 + (A / C)**2 + 2 * theta * mtilda*stilda - 2 * theta * stilda * A / C - 2 * mtilda * A / C) - stilda / (1 - np.dot(theta, np.dot(stilda, gamma)))


alpha_opt_Nominal = np.zeros([N_stocks[3],np.size(eta)])
alpha_opt_Worst = zeros([N_stocks[3],np.size(eta)])

MU_Q5 = np.zeros(np.size(eta),)
CI_MU_Q5 = np.zeros(np.size(eta),)
m_Q5 = np.zeros([np.size(eta), 200])
m_Q5tilda = np.zeros([np.size(eta), 200])
s_Q5tilda = np.zeros([np.size(eta), 200])
m_PTF5 = np.zeros(2, )
s_PTF5 = np.zeros(2, )

m_Q=np.zeros([np.size(eta),200])
alpha_opt_Q5=np.zeros([N_stocks[3],np.size(eta)])
alphaNominal_Q5=np.zeros([N_stocks[3],np.size(eta)])
d=np.zeros([N_stocks[3],1])

MUQ = np.zeros(np.size(eta),)
CI_MUQ= np.zeros(np.size(eta),)
MPT1 = np.zeros(np.size(eta),)
VPT1 = np.zeros(np.size(eta),)

for i in range(0,np.size(eta)):
    gammaM=gamma[i]
    ptf_DV=ptf.Efficient_Frontier_Portfolios(LogReturns_DV,[46],gammaM)
    d,dummy,e,f,g = ptf.Efficient_Frontier_Portfolios.Worst_Case_Portfolio(ptf_DV, gammaM, [eta[i]], 0)
    alpha_opt_Q5[:, i]= np.reshape(dummy,[46,])
    j, dummy,h,m,MPT1[i], VPT1[i] = ptf.Efficient_Frontier_Portfolios.Markovitz_Portfolio(ptf_DV, 0)
    alphaNominal_Q5[:, i]=np.reshape(dummy,[46,])
    MU_Q5[i], CI_MU_Q5[i] = normfit(np.dot(np.transpose(alpha_opt_Q5[:, i]), np.transpose(LogReturns_DV)))
    m_Q5[i, :] = np.linspace(MU_Q5[i] - 2.5 * CI_MU_Q5[i], MU_Q5[i] + 2.5 * CI_MU_Q5[i], 200)
    m_Q5tilda[i,:] = m_Q5[i,:] - theta[i] * S[i] - theta[i] * gamma[i] * S[i]
    MUQ[i], CI_MUQ[i] = normfit(np.dot(np.transpose(alphaNominal_Q5[:, i]), np.transpose(LogReturns_DV)))
    m_Q[i, :] = np.linspace(MUQ[i] - CI_MUQ[i], MU_Q5[i] + CI_MUQ[i], 200)
    alpha_opt_Worst[:,i]=alpha_opt(theta[i], gammaM, S[i])
    m_PTF5[i] = np.dot(np.transpose(alpha_opt_Worst[:, i]),mutilda(alpha_opt_Worst[:,i],theta[i],gammaM))
    s_PTF5[i] = np.dot(np.transpose(alpha_opt_Worst[:, i]),np.dot(sigmatilda(alpha_opt_Worst[:,i],theta[i],gammaM),alpha_opt_Worst[:,i]))


x0 = np.zeros([1,1])
for i in range(0,np.size(eta)):
    for j in range(0,np.size(m_Q5tilda[1,:])):
     Z = sp.optimize.fsolve(function,x0,args=(m_Q5tilda[i,j],theta[i]))
     s_Q5tilda[i,j] = Z[0]



fig67= plt.figure()
plot(s_Q5tilda[0,:], m_Q5tilda[0,:], label="Efficient Frontier")
plt.scatter(s_PTF5[0], m_PTF5[0])
xlabel('Var')
ylabel('E[x]')
title('Efficient Frontier Worst Case for eta=5%')
plt.show()

fig54 = plt.figure()
plot(s_Q5tilda[1,:], m_Q5tilda[1,:], label="Efficient Frontier")
plt.scatter(s_PTF5[1], m_PTF5[1])
xlabel('Var')
ylabel('E[x]')
title('Efficient Frontier Worst Case for eta=5%')
plt.show()

#fig71 = plt.figure()
#plot(s_Q(m_Q), m_Q5, label="Efficient Frontier")
#plt.scatter(VTP1[0], MTP1[0])
#xlabel('Var')
#ylabel('E[x]')
#title('Efficient Frontier Nominal Case for eta=5%')
#plt.show()
#
#fig72 = plt.figure()
#plot(s_Q(m_Q), m_Q5, label="Efficient Frontier")
#plt.scatter(VTP1[1], MTP1[1])
#xlabel('Var')
#ylabel('E[x]')
#title('Efficient Frontier Nominal Case for eta=10%')
#plt.show()

