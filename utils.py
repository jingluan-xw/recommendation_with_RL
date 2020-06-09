import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
import os
import gc
import pickle

if torch.cuda.is_available():
    #print("cuda is available")
    import torch.cuda as t
    device = torch.device('cuda:0')
else:
    #print("cuda is unavailable")
    import torch as t
    device = torch.device('cpu')


def poly2_features(x_vector):
    '''
    Map original features to its polynomial feature vector with power=2, e.g.
    [x1, x2, x3] ---> [1, x1, x2, x3, x1*x2, x2*x3, x1*x3, x1^2, x2^2, x3^2]

    Input:
    ------
    x_vector: 2D torch tensor of shape ([1, num_columns])
              containing the orginal features.

    Output:
    ------
    x_poly2: 1D torch vector of shape ([length])
             containing the polynomial features with power 2.
    '''
    x_2 = x_vector.reshape(-1,1) * x_vector
    num_columns = x_vector.shape[1]
    x_2_triu = x_2[np.triu_indices(num_columns)]
    x_poly2_pre = torch.cat([t.FloatTensor([1]), x_vector.reshape(-1), x_2_triu], dim=0)
    x_poly2 = x_poly2_pre / torch.norm(x_poly2_pre)
    pickle.dump(x_poly2, open("x_poly2.pkl", "wb"))

    x_2 = None
    x_2_triu = None
    x_poly2_pre = None
    x_poly2 = None
    del x_2, x_2_triu, x_poly2_pre, x_poly2
    gc.collect()
    t.empty_cache()

    return None

# Pick an unseen item randomly for each user (row).
def rand_unseen(row):
    return np.random.choice(np.argwhere(row == 0).reshape(-1), 1)


# Xtrue, Xnow, Anow ---> Xnext, Rnext
def get_next(Xtrue, X_now, A_now):
    '''
    Get the next state and reward given the current state, action and ground-truth state

    Input:
    -----
    Xtrue: 2D int matrix, the ground-true preferences of users to items.
    X_now: 2D int matrix, the current revealed preferences of users to items.
    A_now: 1D int vector, the indices of items to be recommended to all users.

    Output:
    -------
    X_next: 2D int matrix, adding users' preferences to newly recommended items to X_now.
    R_next: Float, the mean user's preference to the recommended items.
    '''

    N = Xtrue.shape[0] # number of rows (users)
    #reveal_indices = (np.arange(0,N), np.array(A_now))
    reveal_indices = (t.LongTensor(range(0,N)), A_now.long())
    X_next = X_now
    X_next[reveal_indices] = Xtrue[reveal_indices]
    R_next = Xtrue[reveal_indices].sum() / N
    return X_next, R_next


class Policy():
    '''
    Policy, mappying state to action.
    '''
    def __init__(self, X, K, eps, eta, Wa, only_svd=False):
        self.X = X
        self.K = K
        self.eps = eps
        self.eta = eta
        self.Wa = Wa
        self.only_svd = only_svd
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.S = torch.zeros(K * (self.N + self.M +1), device=device)
        self.A = torch.zeros(self.N, device=device)

    def get_U_Sigma_Vt(self):
        '''
        Get the matrix factorizations due to Truncated SVD.
        '''
        n_iter = 10 * self.K # number of iterations for Truncated SVD.

        # Truncated SVD: X ---> U, Sigma, Vt ---> X_refilled
        U, Sigma, Vt = randomized_svd(self.X.cpu().numpy(), n_components=self.K, n_iter=n_iter)

        return U, Sigma, Vt

    def get_Asvd(self):
        '''
        Get the action due to Truncated SVD algorithm
        '''
        U, Sigma, Vt = self.get_U_Sigma_Vt()
        X_refilled = np.matmul(np.matmul(U, np.diag(Sigma)), Vt)

        # Assign very negative number to items that have been seen by users.
        # So that they won't be picked by the recommender.
        X_refilled[self.X.cpu() != 0] = -1e3

        # find the index of the item with the highest predicted preference for each user (row)
        A_svd = np.apply_along_axis(np.argmax, 1, X_refilled)

        return A_svd

    def fill_S(self):
        '''
        Fill the attribute self.S.
        '''
        # call get_Asvd
        U, Sigma, Vt = self.get_U_Sigma_Vt()

        # scale Sigma
        Sigma_scaled = Sigma / 1e1
        #(Sigma - Sigma.min()) / (Sigma.max() - Sigma.min())

        # Represent the current state by S_now
        S_now = np.concatenate([U.reshape(-1), Sigma_scaled, Vt.reshape(-1)], axis=0)
        self.S = t.FloatTensor(S_now).requires_grad_(True)

    def get_Agreedy(self):
        '''
        Get the greedy action, an improved version of A_svd based on the action-value function.
        '''
        # Call get_S
        A_svd_npfloat = self.get_Asvd()

        # I need dq/dA_greedy, so I shall define A_greedy as a torch.tensor with
        # requires_grad=True.
        A_svd = t.FloatTensor(A_svd_npfloat).requires_grad_(True)
        A_svd_Int = t.LongTensor(A_svd_npfloat)

        # Scaled A_greedy
        A_svd_scaled = A_svd / self.M

        # Put S_now and A_greedy_scaled into a 1D torch.Tensor
        S_A = torch.cat([self.S.reshape(1,-1), A_svd_scaled.reshape(1,-1)], dim=1)

        # Construct a polynomial Feature vector with power 2 for the
        # state and action pair (S_now, A_svd)
        x_2 = S_A.reshape(-1,1) * S_A
        num_columns = S_A.shape[1]
        x_2_triu = x_2[np.triu_indices(num_columns)]
        x_poly2_pre = torch.cat([t.FloatTensor([1]), S_A.reshape(-1), x_2_triu], dim=0)
        S_A_poly2 = x_poly2_pre / torch.norm(x_poly2_pre)
        #-----------------------------------------------------------------------

        # Calculate action-value q(S_now, A_greedy)
        q_Asvd = torch.matmul(self.Wa.reshape(1,-1), S_A_poly2.reshape(-1,1))

        # This could only be done once, once it is done, the graph leading to q_Agreedy is destroyed.
        q_Asvd.backward()

        # dq/dAgreedy, A_greedy.grad could only be called once as well!
        dq_dAsvd = A_svd.grad

        # Improve A_greedy, remember to make sure A_greedy contains integers
        A_greedy = A_svd + self.eta * dq_dAsvd

        # print(f"A_svd == A_greedy before round and long? {np.all((A_svd==A_greedy).cpu().detach().numpy())}")
        # print(f"max |A_svd - A_greedy| == {np.max(np.abs((A_svd-A_greedy).cpu().detach().numpy()))}")
        # print(f"min |A_svd - A_greedy| == {np.min(np.abs((A_svd-A_greedy).cpu().detach().numpy()))}")

        A_greedy = torch.round(A_svd + self.eta * dq_dAsvd).long()
        #print(f"A_greedy.shape = {A_greedy.shape}")
        for i in range(A_greedy.shape[0]):
            # if A_greedy[i] is out of range [0, M-1]
            if (A_greedy[i] > self.M -1) or (A_greedy[i]<0):
                # print("A_greedy[i] is out of range.")
                # print(f"A_svd_Int[i] = {A_svd_Int[i]}")
                A_greedy[i] = A_svd_Int[i]
            # or if A_greedy[i] item has already been recommended.
            elif self.X[i, A_greedy[i]] != 0:
                # print("A_greedy[i] has been recommended before.")
                # print(f"A_svd_Int[i] = {A_svd_Int[i]}")
                A_greedy[i] = A_svd_Int[i]

        pickle.dump(A_greedy, open("A_greedy.pkl", "wb"))


        # # -----------Checking Purpose, does Agreedy really improve q?-----------
        # # Scaled A_greedy
        # A_greedy_scaled = A_greedy.float() / self.M
        # # Put S_now and A_greedy_scaled into a 1D torch.Tensor
        # S_A = torch.cat([self.S.reshape(1,-1), A_greedy_scaled.reshape(1,-1)], dim=1)
        # # Construct a polynomial Feature vector with power 2 for the
        # # state and action pair (S_now, A_greedy)
        # x_2 = S_A.reshape(-1,1) * S_A
        # num_columns = S_A.shape[1]
        # x_2_triu = x_2[np.triu_indices(num_columns)]
        # x_poly2_pre = torch.cat([t.FloatTensor([1]), S_A.reshape(-1), x_2_triu], dim=0)
        # S_A_poly2 = x_poly2_pre / torch.norm(x_poly2_pre)
        # # Calculate action-value q(S_now, A_greedy)
        # q_Agreedy = torch.matmul(self.Wa.reshape(1,-1), S_A_poly2.reshape(-1,1))
        # print(f"eta = {self.eta}")
        # print(f"q_Asvd = {q_Asvd}")
        # print(f"q_Agreedy - q_Asvd = {q_Agreedy.item() - q_Asvd.item()}")
        # print(f"A_svd_Int==A_greedy? {np.all((A_svd_Int==A_greedy).cpu().numpy())}")
        # #-----------------------------------------------------------------------


        # To avoid accumulating usage of memory.
        x_2 = None
        x_2_triu = None
        x_poly2_pre = None
        A_svd_npfloat = None
        A_svd_Int = None
        A_svd = None
        S_A = None
        S_A_poly2 = None
        q_Asvd = None
        dq_dAsvd = None
        A_greedy = None
        del x_2, x_2_triu, x_poly2_pre, A_svd_npfloat, A_svd_Int, A_svd, S_A, \
            S_A_poly2, q_Asvd, dq_dAsvd, A_greedy
        gc.collect()
        t.empty_cache()

        return None

    def get_Arand(self):
        '''
        Get the random action.
        '''
        # This is to make sure that A_rand has the same type as A_greedy
        A_rand = t.IntTensor(np.apply_along_axis(rand_unseen, 1, self.X.cpu()).reshape(-1))

        return A_rand

    def fill_A(self):
        '''
        Choose one action from A_greedy and A_rand by chances 1-eps and eps respectively.
        '''
        if self.only_svd:
            A_svd = self.get_Asvd()
            self.A = t.IntTensor(A_svd).reshape(-1)
        else:
            _ = self.get_Agreedy()
            A_greedy = pickle.load(open("A_greedy.pkl", "rb"))

            if self.eps > 0:
                A_rand = self.get_Arand()

                choose_greedy = np.random.choice([True, False], 1, [1e0-self.eps, self.eps])
                # The unscaled A_now is for updating the next X,
                # since A_now contains the indices for items to recommend.
                if choose_greedy:
                    A_now = A_greedy
                else:
                    A_now = A_rand

                # Assign the attribute
                self.A = A_now

            elif self.eps == 0e0: # self.eps == 0
                self.A = A_greedy
            else:
                print("eps should be >=0, but get eps <0.")

            # #--------Check whether self.A == A_svd when eta=0 and eps=0---------
            # A_svd = self.get_Asvd()
            # A_svd_Int = t.IntTensor(A_svd).reshape(-1)
            # print(f"self.A == A_svd? {torch.all(torch.eq(self.A, A_svd_Int))}")
            # # Okay I find that sometime they are equal and other times they are
            # # not, and the two cases happen randomly. There must be some random
            # # error happening.
            # #-------------------------------------------------------------------

    def get_state_poly(self):
        '''
        Get the polynomial feature with power=2 for the state.
        '''
        _ = poly2_features(self.S.reshape(1,-1))
        S_poly2 = pickle.load(open("x_poly2.pkl", "rb"))

        return S_poly2

    def get_state_action_poly(self):
        '''
        Get the polynomial feature with power=2 for the (state, action) pair.
        '''
        A_now_scaled = self.A.float() / self.M

        # Put S_now and A_greedy_scaled into a 1D torch.Tensor
        S_A = torch.cat([self.S.reshape(1,-1), A_now_scaled.reshape(1,-1)], dim=1)

        # Construct a polynomial Feature vector with power 2 for the state and action pair
        _ = poly2_features(S_A)
        S_A_poly2 = pickle.load(open("x_poly2.pkl", "rb"))

        return S_A_poly2
