import numpy as np
from scipy.special import comb
facts = np.ones(100+1) # CAUTION, precompute factorials, but set to a max of n=100
for i in range(1,len(facts)):
  facts[i] = facts[i-1]*i
facts

def top_k_to_mat(perm):
    n = len(perm)
    k = len(perm[~np.isnan(perm)])
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):    
            if not np.isnan(perm[i]) and not np.isnan(perm[j]): mat[i,j] = perm[i]<perm[j]
            elif  np.isnan(perm[i]) and  np.isnan(perm[j]):     mat[i,j] = .5
            elif  np.isnan(perm[i]) and not np.isnan(perm[j]):  mat[i,j] = 0
            elif  not np.isnan(perm[i]) and np.isnan(perm[j]):  mat[i,j] = 1
            mat[j,i] = 1-mat[i,j]
    return mat
    
    
def p_rank_to_mat(perm):
    # in this case and in p_rank_to_mat_GENERATION the NaNs are *not observed*
    # they can be ranked befor the items already seen
    #for top-k rankings see 
    # if this throws an error because of missing params , return_ratios=True, return_num_comparisons=False
    # then change call to p_rank_to_mat_GENERATION
    n = len(perm)
    k = len(perm[~np.isnan(perm)])
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):    
            if not np.isnan(perm[i]) and not np.isnan(perm[j]): mat[i,j] = perm[i]<perm[j]
            elif  np.isnan(perm[i]) and  np.isnan(perm[j]):     mat[i,j] = .5
            elif  np.isnan(perm[i]) and not np.isnan(perm[j]):  mat[i,j] = (perm[j]+1)/(k+1)
            elif  not np.isnan(perm[i]) and np.isnan(perm[j]):  mat[i,j] = 1-(perm[i]+1)/(k+1)
            mat[j,i] = 1-mat[i,j]
    return mat

# k the fixed (partial) ranking
# n-k the rest, the unknown
# TODO precomput the next 3 functions in a table to speed up
def V(n,m):
  return facts[m]/facts[m-n]
def S(m,n): #show many ways of shuffling two piles of this lengths
  #return facts[n+m]/(facts[n]*facts[m])
    return comb(n + m, n, exact=True) 
def perms_total(n,k):
  return facts[n-k] * S(k,n-k)


def perms_with_nan_before(n,k,p_rank):
  # given k partially ordered items
  # how many permutations of n items are consistent with the k
  # and place one of the unknown items BEFORE the item with p_rank partial rank
  acum = 0
  for i in range(0,n-k): # aprate de ese q ya has cogido, cuanto mas coges para poner delante
    acum += V(i, n-k-1) * (i+1) * S(p_rank, i+1) * facts[(n-k-i-1)] * S(n-k-i-1, k-p_rank-1)
  return acum

def p_rank_to_mat_GENERATION(perm, return_ratios=True, return_num_comparisons=False):
    # use p_rank_to_mat if you want to output the Pmatrix only
  n = len(perm)
  p = np.zeros((n,n))
  compared = np.zeros((n,n))
  k = (~np.isnan(perm)).sum()
  for i in range(n):
    for j in range(i+1,n):
      if perm[i] < perm[j]:
        p[i,j] = perms_total(n,k)
        compared[i,j] += 1
        compared[j,i] += 1
      elif perm[j] < perm[i]:
        p[j,i]=perms_total(n,k)
        compared[i,j] += 1
        compared[j,i] += 1
      elif  np.isnan(perm[i]) and np.isnan(perm[j]):
        p[j,i] = perms_total(n,k)/2
        p[i,j] = perms_total(n,k)/2
      elif np.isnan(perm[j]) :
        p_rank = int(perm[i])
        p[j,i] = perms_with_nan_before(n,k,p_rank)# / perms_total(n,k)
        p[i,j] = perms_total(n,k) - p[j,i]
      else:
        p_rank = int(perm[j])
        p[j,i] = (perms_total(n,k)-perms_with_nan_before(n,k,p_rank)) #/ perms_total(n,k)
        p[i,j] = perms_total(n,k) - p[j,i]
  if return_ratios: p = p/perms_total(n,k)
  if return_num_comparisons: return p, compared, perms_total(n,k)
  return p



def borda_mat(p):
#   print(p)
#   print(p.sum(axis=0))
  return np.argsort(np.argsort(p.sum(axis=0)))
