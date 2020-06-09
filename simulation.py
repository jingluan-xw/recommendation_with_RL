import numpy as np
import torch
from utils import *
import argparse
import pickle
import gc
import os

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    print("cuda is available")
    import torch.cuda as t
    device = torch.device('cuda:0')
else:
    print("cuda is unavailable")
    import torch as t
    device = torch.device('cpu')


# This allows one to input hyperparameters without modifying this script file.
parser = argparse.ArgumentParser()

# args.policy='svd' if only use svd, otherwise will use the svd helped with RL.
parser.add_argument("--policy", type=str, help="only svd or the svd_with_RL.")
# If include "--LoadWeights" in the terminal, then args.LoadWeights=True. If not type
# "--LoadWeights" in the terminal, then args.LoadWeights=False.
parser.add_argument("--LoadWeights", action="store_true")
parser.add_argument("--N", default=1000, type=int, help="number of users")
parser.add_argument("--M", default=100, type=int, help="number of items")
parser.add_argument("--K", default=5, type=int, \
                    help="number of singular values (eigen values) to retain in Truncated SVD.")
parser.add_argument("--eps", default=0.1, type=float, \
                    help="probability for choosing the random action")
parser.add_argument("--eta", default=0.1, type=float, help="improvement rate")
parser.add_argument("--alpha", default=0.01, type=float, \
                    help="learning rate for Reinforcement learning")
parser.add_argument("--gamma", default=0.3, type=float, \
                    help="discounting factor for future return")
parser.add_argument("--startfrac", default=0.1, type=int, \
                    help="starting fraction of known preferences per user")
parser.add_argument("--Mrecom", default=30, type=int, help="number of recommendations per episode")
parser.add_argument("--startepi", default=0, type=int, help="starting episode number")
parser.add_argument("--Nepi", default=30, type=int, help="max number of episodes")
parser.add_argument("--thresh", default=0.0001, type=float, \
                    help="threshold for convergence of weights in the value functions.")


args = parser.parse_args()

# --------- Define Hyperparameters  --------------------------
N = args.N # number of users
M = args.M # number of items
K = args.K # number of singular values (eigen values) to retain in Truncated SVD.
eps = args.eps # probability for choosing the random action
eta = args.eta # improvement rate
# learning rate for Reinforcement learning v <--- (1-alpha)* v + alpha * target_v
alpha = args.alpha
# discounting factor for future return, return_t = immediate_reward + gamma * return_{t+1}
gamma = args.gamma
# We initialize by revealing a fraction of preferences to items per user
start_frac = args.startfrac
# number of recommendations per episode
# In real life, a recommender faces a continuing task. For simplicity of this project, we
# artificially define an "episode" during which the recommender interacts with users
# by M_recom times.
M_recom = args.Mrecom
# number of episodes in this simulation.
n_epi = args.Nepi
# the threshold for judging the convergence of weights.
# If the L2-norm of the concatenated weights of the state and action value functions
# becomes smaller than this threshold, the simulation stops repeating the episode.
thresh = args.thresh
if args.policy == 'svd':
    only_svd = True
elif args.policy == 'svdRL':
    only_svd = False
else:
    print('''Please input either 'svd' or 'svdRL' for the '--policy' argument.
          Will adopt 'svdRL' for now.''')
    only_svd = False

print(f"only_svd = {only_svd}")

# Initialize the weights in action-value function, Wa, to all zeros.
n_S = K * (N + M + 1)
n_SA = K * (N + M + 1) + N
len_Ws = int(n_S**2/2 + 3*n_S/2 + 1)
len_Wa = int(n_SA**2/2 + 3*n_SA/2 + 1)

if args.LoadWeights:
    Ws = pickle.load(open("Ws.pkl", "rb"))
    Wa = pickle.load(open("Wa.pkl", "rb"))
    start_episode = args.startepi
else:
    Ws = torch.zeros(len_Ws, device=device)
    Wa = torch.zeros(len_Wa, device=device)
    start_episode = 0

# `Xtrue` is the ground-truth matrix for all users' preferences to all items.
# It is known by the Monte-Carlo Simulator but not known by the recommender.
# `Xtrue` is generated randomly such that its element is 1 by 50% chance and
# -1 by 50% chance.
Xtrue = torch.rand(N, M, device=device)
Xtrue[Xtrue >= 0.5] = 1
Xtrue[Xtrue < 0.5] = -1

# We initialize X_now by revealing the ground-truth preferences to M_start items for each user
M_start = int(start_frac * M)

# -------------------- Choose X_init randomly------------------------------------
X_init = torch.zeros(N, M, device=device)
# Pick randomly M_start items to reveal for each user.
for i in range(0, N):
    rand_index = np.random.choice(range(0,M), size=M_start, replace=False)
    X_init[i, rand_index] = Xtrue[i, rand_index]

# Initialize the number of episodes and the change of weights.
episode = start_episode
W_change = thresh * 10

# Keep a record of the returns after M_recom times of interactions between the
# recommender and users. We define the return as the cummulative user-average reward
# without discounting. Following the denotions in Sutton and Barto, we use "G" to
# label the return.


while (episode < n_epi + start_episode) and (W_change > thresh):
    '''
    Repeat episodes until when maximum number of episodes is reached
    or when the weights of value functions converge.
    '''
    # Remember the old weights before the new episode starts.
    W_old = torch.cat([Wa, Ws])

    print(f"episode = {episode+1}: beginning")
    # -------------------- Initialize X_now ------------------------------------
    X_now = X_init # every episode starts with the same initial X.

    reward_list = []
    G_no_discount = 0e0
    # initialize the return to 0 at the beginning of each episode.
    for recom in range(M_recom):
        '''
        Simulate user-recommender interations per episode.
        '''

        # ----------------- Policy Improvement (Current state) -----------------
        # pi(S_t): the policy based on the current state, X_now.
        # print(f"Wa = {Wa}")
        Pi_now = Policy(X_now, K, eps, eta, Wa, only_svd)
        Pi_now.fill_S()
        Pi_now.fill_A()
        # get the current action (improved version)
        A_now = Pi_now.A
        # get the polynomial feature vections for the state and the (state, action) pair.
        S_poly2 = Pi_now.get_state_poly()
        S_A_poly2 = Pi_now.get_state_action_poly()


        # ------------ Monte Carlo Simulation of the user's response -----------
        # Take action A_now, generate the next state, the immediate reward.
        X_next, R_next = get_next(Xtrue, X_now, A_now)

        # Accumulate the immediate return at every interation between the recommender
        # and the users.
        reward_list.append(R_next.item())
        G_no_discount += R_next.item()

        # ----------------- Policy Improvement (Next state) --------------------
        # pi(S_{t+1}): the policy based on the next state, X_next.
        Pi_next = Policy(X_next, K, eps, eta, Wa, only_svd)
        Pi_next.fill_S()
        Pi_next.fill_A()
        # get the polynomial feature vections for the state and the (state, action) pair.
        S_poly2_next = Pi_next.get_state_poly()
        S_A_poly2_next = Pi_next.get_state_action_poly()


        #--------------- Policy Evaluation ---------------------------------------------
        # Update Ws, weights for the state-value function.
        Ws_Delta_S = torch.mm((S_poly2 - gamma*S_poly2_next).reshape(1,-1), Ws.reshape(-1,1)).reshape(-1)
        Ws = Ws + alpha * (R_next - Ws_Delta_S) * S_poly2

        # Update Wa, weights for the action-value function.
        Wa_Delta_SA = torch.mm((S_A_poly2 - gamma*S_A_poly2_next).reshape(1,-1), Wa.reshape(-1,1)).reshape(-1)
        Wa = Wa + alpha * (R_next - Wa_Delta_SA) * S_A_poly2


        os.system("rm X_next.pkl")
        pickle.dump(X_next, open("X_next.pkl", "wb"))
        os.system("rm Ws.pkl")
        pickle.dump(Ws, open("Ws.pkl", "wb"))
        os.system("rm Wa.pkl")
        pickle.dump(Wa, open("Wa.pkl", "wb"))

        Ws = None
        Wa = None
        X_next = None
        Ws_Delta_S = None
        Wa_Delta_SA = None
        S_poly2 = None
        S_poly2_next = None
        S_A_poly2 = None
        S_A_poly2_next = None
        Pi_now = None
        Pi_next = None
        A_now = None

        gc.collect()
        t.empty_cache()

        X_now = pickle.load(open("X_next.pkl", "rb"))
        Ws = pickle.load(open("Ws.pkl", "rb"))
        Wa = pickle.load(open("Wa.pkl", "rb"))

        # # Monitor memory usage.
        # os.system("gpustat")
        # os.system("free -m ") # cpu ram

        if (recom + 1) % 10 == 0:
            print(f"---- recom: {recom+1}, G_no_discount = {G_no_discount}")


    # Update episode (number of episodes) and the change of the weight.
    episode += 1
    W_new = torch.cat([Wa, Ws])
    W_change = torch.norm(W_old - W_new)

    print(f"episode = {episode}, G_no_discount = {G_no_discount}, W_change = {W_change}")

    # End of an episode, store the history of immediate rewards.
    pickle.dump(reward_list, \
                open("../Output/reward_"+args.policy+"_"+str(episode)+".pkl", "wb"))

# pickle.dump(G_list, open("../Output/G_list_"+args.policy+".pkl", "wb"))
pickle.dump(Wa, open("../Output/Wa_"+args.policy+".pkl", "wb"))
pickle.dump(Ws, open("../Output/Ws_"+args.policy+".pkl", "wb"))

# Keep a copy of the latest weights in the Code dir.
pickle.dump(Wa, open("Wa_"+args.policy+".pkl", "wb"))
pickle.dump(Ws, open("Ws_"+args.policy+".pkl", "wb"))
