import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import pickle

from itertools import combinations
np.random.seed(7)


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


def sigmoid_cal(theta,context_vector,K):
    # takes in theta and and design matrix as array of vector
    # returns mu_i vector
    sigmoid = np.zeros(K)
    sigmoid_return=np.zeros(K)
    
    [a] = theta.shape
    sigmoid = np.exp(np.dot(theta,context_vector))
    
    sigmoid_return= sigmoid/(1+np.sum(sigmoid))
    
    return sigmoid_return


def find_best_assortment(theta, contexts, K):
  # context is d X N
  all_assortments=np.array(list((combinations(range(N),K))))

  all_assortments_contexts = contexts[:,all_assortments]
  [dim, num, cardinality] = all_assortments_contexts.shape
  
  sigmoid_vals=np.zeros(num)

  for i in range(num):
    sigmoid_vals[i] = np.sum(sigmoid_cal(theta,all_assortments_contexts[:,i,:],K))

  
  max_index = np.argmax(sigmoid_vals)

  max_assortment =all_assortments_contexts[:,max_index,:]

  return max_assortment


def log_loss_cal(theta,data):
  # T is current time step, we calculate log loss till t-1
  # theta is parameter
  # context matrix is the design matrix till t
  # lbd is the lambda parameter

  [t,context_matrix,lbd,K,theta_star, reward_val] = data

  log_loss=np.zeros(t-1)
  for i in range(0,t-1):
    sigmoid_values_current = sigmoid_cal(theta,context_matrix[:,:,i],K)

    sigmoid_values_true = sigmoid_cal(theta_star,context_matrix[:,:,i],K)

    log_loss[i] = np.sum(np.dot(np.transpose(reward_val[:,i]),np.log(sigmoid_values_current)))

  log_loss_return = np.sum(log_loss) -lbd*np.dot(theta,theta)

  return log_loss_return


def gamma_func(lbd,t,K):
  gamma_val = np.sqrt(lbd)/2+(2/np.sqrt(lbd))*np.log(((np.power((lbd+L*K*t/d),d/2)*np.power(lbd,-d/2))/delta)) + 2*d/np.sqrt(lbd)*np.log(2)

  return gamma_val

def beta_func(lbd,t,K):
  gamma_val = gamma_func(lbd,t,K)
 
  beta_val= gamma_val+(gamma_val*gamma_val)/lbd
  return beta_val


def confidence_set(theta_est,theta_base_set,lbd,t,K,context_matrix,reward_val):
  theta_in_conf_set =[]

  [dim, base_size]= theta_base_set.shape


  beta_val = beta_func(lbd,t,K)
  

  data = [t,context_matrix,lbd,K,theta_star,reward_val]

  log_loss_call_est = log_loss_cal(theta_est,data)
  

  for i in range(0,base_size):
    if log_loss_cal(theta_base_set[:,i],data) <= beta_val*beta_val + log_loss_call_est:
      
      theta_in_conf_set.append(theta_base_set[:,i])
  
  return np.array(theta_in_conf_set)


def find_best_assortment_for_all_theta(theta_in_confidence_set,contexts,K):

  max_val=0
  max_assortment =np.zeros(K)
  best_theta = np.zeros(d)
  [theta_in_confidence_set_size, dim]= theta_in_confidence_set.shape
  for i in range(0,theta_in_confidence_set_size):
    theta = theta_in_confidence_set[i]
    current_assortment = find_best_assortment(theta,contexts,K)
    current_assortment_val = np.sum(sigmoid_cal(theta,current_assortment,K))
    if current_assortment_val > max_val:
      max_val=current_assortment_val
      max_assortment = current_assortment_val
      best_theta=theta
    
  
  return [best_theta,max_assortment]


# find ML estimate theta

# minimize negative of log loss

def log_loss_mle(data):

  res= opt.minimize(
      fun =lambda theta, data: -log_loss_cal(theta,data),
      x0= np.zeros(d), args=(data,), method='BFGS'
  )

  return res.x

def play_assortment(assortment,K,theta_star):
  sigmoid_vals = sigmoid_cal(theta_star,assortment,K)
  multinomial_params = np.zeros(K+1)
  multinomial_params[0] = 1-np.sum(sigmoid_vals)
  reward_generated = np.random.multinomial(1,multinomial_params)

  return reward_generated[1:K+1]

mc_runs = 200
d=1
T = 15
K=1
theta_base_set_size =8
delta=0.1
theta_star = np.array([.1])
N=5 # number of items in the inventory
cum_regret_mc_runs = np.zeros((mc_runs,T))

reward_coeff = np.ones(K+1)
reward_coeff[0] = 0

all_contexts = np.random.uniform(low=-1.0,high=1.0,size =(d,N))

np.random.seed(7)


for mc in range(mc_runs):    
    reward_matrix = np.zeros((d,T))

    L=1 #upper bound on the lipchitz constant of softmax

    all_contexts_time_varying = np.random.uniform(low=-1.0,high=1.0,size =(d,T,N))


    theta_base_set = np.random.uniform(low=-1.0,high=1.0,size=(d,theta_base_set_size))

    regret =np.zeros(T)

    accrued_rewards= np.zeros((K,T))

    contexts_chosen = np.zeros((d,K,T))

    lbd = d*np.log(K*T)


    t=0
    optimal_assortment = find_best_assortment(theta_star,all_contexts_time_varying[:,t,:],K)
    optimal_sigmoid_val = sigmoid_cal(theta_star,optimal_assortment,K)
    optimal_expected_reward_val = np.sum(optimal_sigmoid_val)

    theta_est =np.zeros(d)
    theta_in_confidence_set =np.transpose(theta_base_set)
    a=find_best_assortment_for_all_theta(theta_in_confidence_set,all_contexts_time_varying[:,t,:],K)
    best_theta=a[0]
    best_assortment_found =a[1]
    contexts_chosen[:,:,t] = best_assortment_found
    accrued_rewards[:,t] = play_assortment(best_assortment_found,K,theta_star)
    accrued_expected_reward_val =np.sum(sigmoid_cal(theta_star,best_assortment_found,K))
    regret[t]=optimal_expected_reward_val-accrued_expected_reward_val


    for t in range(1,T):
        optimal_assortment = find_best_assortment(theta_star,all_contexts_time_varying[:,t,:],K)
        optimal_sigmoid_val = sigmoid_cal(theta_star,optimal_assortment,K)
        optimal_expected_reward_val = np.sum(optimal_sigmoid_val)

        data = [t,contexts_chosen,lbd,K,theta_star,accrued_rewards]
        theta_est = log_loss_mle(data)
        theta_in_confidence_set = confidence_set(theta_est,theta_base_set,lbd,t,K,contexts_chosen,accrued_rewards)
  
        a= find_best_assortment_for_all_theta(theta_in_confidence_set,all_contexts_time_varying[:,t,:],K)
        best_theta =a[0]
        best_assortment_found=a[1]
        contexts_chosen[:,:,t]=best_assortment_found
        accrued_rewards[:,t] = play_assortment(best_assortment_found,K,theta_star)
        accrued_expected_reward_val = np.sum(sigmoid_cal(theta_star,best_assortment_found,K))
        regret[t] = optimal_expected_reward_val-accrued_expected_reward_val
  
  
    cum_regret =np.zeros(T)
    for i in range(0,T):
        cum_regret[i] = cum_regret[i]+regret[i]
    
    
    
    cum_regret_mc_runs[mc,:] = cum_regret

pickling_on = open("EJOR_convex_relaxed","wb")
    
pickle.dump(cum_regret_mc_runs, pickling_on) 