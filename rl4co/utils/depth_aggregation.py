import partial_rankings as pr
import numpy as np
import itertools as it
import mallows_kendall as mk
import mallows_hamming as mh
import permutil as pu
import math
from scipy.spatial.distance import mahalanobis
import scipy


def bordaP(P):
  scores = np.nan_to_num(P,0).sum(axis=0)
  return np.argsort(np.argsort(scores))

#     sns.heatmap(P,ax=ax[1])
def Borda(rankings):
  return np.argsort(np.argsort(rankings.sum(axis=0)))
# f, ax = plt.subplots(1,3, figsize=(7,4))
# plot_path(P, dfpath,sample, dist_name_depth,ax=ax[1])


def is_in_path(sigma, sigma1, sigma2):
    if sigma.shape != sigma2.shape:
        print(sigma.shape, sigma2.shape)
        print(sigma, sigma2)
    mask = sigma1==sigma2
    res = int((sigma2[mask] == sigma[mask]).all())
    #equivalently
    #s  = (sigma1+sigma2)/2 + sigma
    #res_slow = int((s==1).sum() == 0)
    #assert(res==res_slow)
    return res
                          
#def get_depth_prank(prankm,P): DONE BELOW
 #   return (prankm*P).sum()

def get_depth(P, perm, depth_type='distance', perm_mat=None, sample_distances=None, 
              sample_matrices=None, VI=None, sample=None,perm_ind=None):
    #if type(perm) != int: n = len(perm)
    n = len(perm)
    if  depth_type == 'distance': 
        return pu.max_dist(n,'k') - pu.dist_to_sample(perm,P,dist_name='k')
        # np.tril(P[np.ix_(np.argsort(perm),np.argsort(perm))],k=-1).sum() <- dist2sample
    if  depth_type == 'hamming': 
        return pu.max_dist(n,'h') - pu.dist_to_sample(perm,P,dist_name='h')
    if  depth_type == 'prod':
        inv = np.argsort(perm)
        aux = np.triu(P[np.ix_(inv, inv)],k=1)
        return np.prod(aux[np.triu_indices(n,1)])
    if depth_type == 'halfspace':
        inv = np.argsort(perm)
        aux = np.triu(P[np.ix_(inv, inv)],k=1)
        return aux[aux!=0].min() 
    if depth_type == 'lens_path':
        assert(sample_matrices is not None)#reference sample
        #assert(perm_mat is not None)#test sample
        if perm_mat is None: perm_mat = pu.sample_to_marg(np.array([perm]))
        #assert(perm_ind is not None)# if the sample is in references sample, its index
        if perm_ind is None:
            perms_ind=-1
            perm_mat = [pu.sample_to_marg(np.array([perm]))][0]
        c = 0 
        m = len(sample_matrices)
        
        for i,j in it.combinations(range(m),2):
            if i != perm_ind and j != perm_ind: 
                #sigma1, sigma2 = sample_matrices[i],sample_matrices[j]
                c += is_in_path(perm_mat, sample_matrices[i],sample_matrices[j])
        return c/(m*(m-1)/2)
    if depth_type == 'lens_path_full': #same as 'lens_path'
        assert(sample_matrices is not None)#reference sample
        if perm_mat is None: perm_mat = pu.sample_to_marg(np.array([perm]))
        c = 0 
        m = len(sample_matrices)
        for i,j in it.combinations(range(m),2):
            c += is_in_path(perm_mat, sample_matrices[i],sample_matrices[j])
        return c/(m*(m-1)/2)
    if depth_type == 'lens_precomputed': 
        # the distances among all pairs of permus have to be precimputed in sample_distances
        # the perm for which the depth is being computed has to be IN the sampel
        assert(sample_distances is not None)
        assert(perm_ind is not None)
        m = len(sample_distances)
        #cc = 0 
        #for i, j in it.combinations(range(len(sample_distances)),2):
        #        d = sample_distances[i,j]
        #        if sample_distances[i,perm_ind] <= d and sample_distances[j,perm_ind] <= d: 
        #            cc += 1
        dist_to_perm = sample_distances[:, perm_ind]
        indices = np.array(list(it.combinations(range(m), 2)))
        d = sample_distances[indices[:, 0], indices[:, 1]]
        mask = (dist_to_perm[indices[:, 0]] <= d) & (dist_to_perm[indices[:, 1]] <= d)
        c = np.sum(mask)
        #assert(cc==c)
        return c/(m*(m-1)/2)
    if depth_type == 'lens': 
        #TODO: for the DA algorithm, since perms that are evaluated consecutively
        # are at distance 1 from each other, this can be speeded up
        assert(sample is not None)
        m = len(sample)
        c = 0 
        for i, j in it.combinations(range(len(sample)),2):
            sigma1, sigma2 = sample[i], sample[j]
            d = mk.distance(np.array(sigma1), np.array(sigma2)) 
            if mk.distance(perm, sigma1) <= d and mk.distance(perm, sigma2) <= d :
               c += 1
        return c/(m*(m-1)/2) 
    if depth_type == 'mahalanobis':
        return -mahalanobis(perm , bordaP(P), VI)
            
        
        





################################################################################################
###############             DEPTH BASED ON DISTANCES AND BTL             #######################
################################################################################################

def depth_aggregation(P, initial_permu=None, test_mode=False, increasing=True, depth_type='distance', sample=None, stochastic_accept=0,sample_matrices=None):
  
    #distances_sample used for the lens depth
    # increasing : we are looking for the deepest permu
  n = P.shape[0]
  res = []
  central = np.arange(n)
  maxiter=50*pu.max_dist(n, 'k')
  #for num_path in range(num_paths):
  dist_vals, pairs, path = [], [], []
  contiter, jumps = 0 , 0
  if initial_permu is None:
      perm = np.random.permutation(range(n))#perm_ini.copy()
      initial_permu = perm.copy()
  else: perm = initial_permu.copy()
  inv = np.argsort(perm)
  depth = get_depth(P, perm, depth_type, sample=sample,sample_matrices=sample_matrices)
  #if depth_type=='distance': depth = pu.max_dist(n, 'k') - pu.dist_to_sample(perm,P, 'k')
  halt = False
  while not halt and contiter<maxiter:#
    # if depth_type=='lens':print("INFO",depth_type,contiter, perm, depth)
    contiter+=1
    perm, inv, depth = find_neigs(P, perm, inv, depth, increasing=increasing,depth_type=depth_type,
                                  sample=sample, stochastic_accept=stochastic_accept,sample_matrices=sample_matrices)
    if perm is None : halt=True
    else:
        res.append([perm.copy(),depth])
        if test_mode:
            if depth_type=='distance' : 
                assert(pu.max_dist(n, 'k') - pu.dist_to_sample(perm,P, 'k') - depth < .00000001)
            if depth_type=='prod' : 
                assert(get_depth(P, perm, depth_type) - depth < .00000001)
  if contiter == maxiter: print("öjooooo MAX ITER reached")
  if len(res)==0:
      return initial_permu, pu.max_dist(n, 'k') - pu.dist_to_sample(initial_permu,P, 'k'), []
  return res[-1][0], res[-1][1], res # => perm, depth, res


def find_neigs(P, perm, inv, depth, method='first', increasing=True, depth_type='distance', sample=None, stochastic_accept=0,sample_matrices=None):
    #distances_sample is for the lend
  n = P.shape[0]
  assert(method=='first') #TODO: the 'best' option
  for i_pair in np.arange(n-1):
    i,j = inv[i_pair],inv[i_pair+1]
    if  depth_type=='distance': depth2 = depth-P[i,j]+P[j,i]
    elif  depth_type=='prod': depth2 = depth/P[i,j]*P[j,i]
    elif  depth_type=='hamming': #TODO check
        depth2 = depth+(1-P[i,perm[i]])+ (1-P[j,perm[j]]) - (1-P[i,perm[j]]) - (1-P[j,perm[i]])
    elif depth_type=='lens' or depth_type=='halfspace' or depth_type=='lens_path':  #TODO speed up
        perm2 = perm.copy()
        perm2[i],perm2[j] = perm2[j],perm2[i]
        depth2 = get_depth(P, perm2, depth_type, sample=sample,sample_matrices=sample_matrices)   
    else : print("depth_aggregation > find_neigs . not done")    
    # if the var increasing is set to true, we go to a new perm if the depth is larger
    if increasing: cond = depth2 > depth#
    else : cond = depth2 < depth
    if cond or np.random.random() < stochastic_accept :
      perm[i],perm[j] = perm[j],perm[i]
      inv[i_pair],inv[i_pair+1] = inv[i_pair+1],inv[i_pair]
      # print("test",depth2,get_depth(P, perm, depth_type=depth_type)-depth2)
      return perm, inv, depth2
  return None, None , None


import torch

def MCMC(P, len_mcmc=10000, burn_out=None, depth_type='distance', test_mode=False):
    # ensure P is a float tensor
    if not torch.is_tensor(P):
        P = torch.tensor(P, dtype=torch.float32)
    else:
        P = P.clone().float()
    device = P.device
    # keep a NumPy copy for get_depth
    P_np = P.cpu().detach().numpy()

    n = P.shape[0]
    if burn_out is None:
        burn_out = n * n

    # initialize a random perm (torch) and its depth via NumPy-based get_depth
    perm = torch.randperm(n, device=device)
    perm_np = perm.cpu().numpy()
    depth_val = get_depth(P_np, perm_np, depth_type)
    depth = torch.tensor(depth_val, dtype=P.dtype, device=device)

    inv = torch.argsort(perm)

    depths = []
    perms = []

    total_steps = burn_out + len_mcmc
    for step in range(total_steps):
        # pick a random adjacent‐pair in the current perm
        i_pair = torch.randint(0, n - 1, (), device=device).item()
        i = inv[i_pair].item()
        j = inv[i_pair + 1].item()

        # propose swapping positions i and j
        if depth_type == 'distance':
            depth_ten = depth - P[i, j] + P[j, i]
        elif depth_type == 'hamming':
            depth_ten = (
                depth
                + (1 - P[i, perm[i]]) + (1 - P[j, perm[j]])
                - (1 - P[i, perm[j]]) - (1 - P[j, perm[i]])
            )
        else:
            depth_ten = depth * (P[j, i] / P[i, j])
            # catch numerical issues by recomputing via get_depth
            if torch.isnan(depth_ten) or depth == 0 or depth == float('inf'):
                tmp = perm[i].clone()
                perm[i], perm[j] = perm[j], tmp
                perm_np = perm.cpu().numpy()
                dval = get_depth(P_np, perm_np, depth_type)
                depth_ten = torch.tensor(dval, dtype=P.dtype, device=device)
                # swap back
                perm[j], perm[i] = perm[i], tmp

        # Metropolis–Hastings acceptance
        rand_val = torch.rand((), device=device).item()
        threshold = float((depth_ten / depth).item())
        cond1 = (depth_type != 'hamming') and (P[i, j] < P[j, i]).item()

        if cond1 or rand_val <= threshold:
            # accept
            depth = depth_ten
            tmp_perm = perm[i].clone()
            perm[i], perm[j] = perm[j], tmp_perm

            tmp_inv = inv[i_pair].clone()
            inv[i_pair], inv[i_pair + 1] = inv[i_pair + 1], tmp_inv

            if test_mode:
                perm_np_check = perm.cpu().numpy()
                dval_check = get_depth(P_np, perm_np_check, depth_type)
                depth_check = torch.tensor(dval_check, dtype=P.dtype, device=device)
                assert torch.allclose(depth_check, depth)

        # after burn‐in, record
        if step >= burn_out:
            depths.append(depth.clone())
            perms.append(perm.clone())

    # stack into tensors
    if depths:
        depths_tensor = torch.stack(depths)
        perms_tensor = torch.stack(perms)
    else:
        depths_tensor = torch.empty((0,), dtype=P.dtype, device=device)
        perms_tensor = torch.empty((0, n), dtype=perm.dtype, device=device)

    return depths_tensor, perms_tensor



################################################################################################
###############                 DEPTH BASED ON DISTANCES                 #######################
################################################################################################
# the next funcitons are for the distribution based on the depth
# analysis in notebook 11

def Z_depth_distri(P): #normalizing constant of the distri based on depths
  n = P.shape[0]
  #aux = (np.full((n,n), 0.5) * P)
  #np.fill_diagonal(aux, 0)
  #Z = np.math.factorial(n)*(aux.sum()) its independent of P, only depends on n
  return math.factorial(n)*n*(n-1)/4 
  #return Z

def depth_distri_marg(M): #still O(n^3) for the row and col vars
  # given a matrix of pairwise distris (relative marginals)
  # contruct the depth-distri D and return the marginals of D
  n = len(M[0,:])
  marg = np.zeros((n,n))
  c1 = 1/24 * n * ( n**2 - 3*n + 2) * (8*(n-2)*math.factorial(n-3) + 3*(n-3)*math.factorial(n-2))
  c2 = math.factorial(n-3) *1/6* n* (n**2 - 3*n +2)
  c3 = math.factorial(n)/2 # n*(n-1)/2*np.math.factorial(n-2)
  Z = Z_depth_distri(M)
  for i in range(n):
    for j in range(i+1,n):
      aux = M[i,:]
      aux = aux[np.arange(n)!=j]
      row = aux[~np.isnan(aux)].sum()
      aux = M[:,j]
      aux = aux[np.arange(n)!=i]
      col = aux[~np.isnan(aux)].sum()
      marg[i,j] = 1/Z * (c1 + c2*row + c2*col + c3*M[i,j])
      marg[j,i]  = 1 - marg[i,j]
  return marg

def depth_distri_marg_pos(P, i,j):
    # different way of computing the marginal, based on the partial rankings
    n = P.shape[0]
    prank = np.array([np.nan]*n)
    prank[i]=0
    prank[j]=1
    prankm,_,prankl = pr.p_rank_to_mat(prank,return_num_comparisons=True)
    return (prankm*P).sum()*(math.factorial(n)/2/Z_depth_distri(P))

def get_depth_prank(P,prankm):
    return (prankm*P).sum()

################################################################################################
####################                 HALSPACE DEPTH            #################################
################################################################################################











#
