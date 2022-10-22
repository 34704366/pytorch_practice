import numpy as np
import argparse
import logging

import pandas as pd

global logger

from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
from tqdm import *

NR_min = 10101001
NMAE_min = 10101001


class BTMF(object):

    def __init__(self, args):
        self.args = args

    def mvnrnd_pre(self, mu, Lambda):
        src = normrnd(size=(mu.shape[0],))
        return solve_ut(cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
                        src, lower=False, check_finite=False, overwrite_b=True) + mu

    def cov_mat(self, mat, mat_bar):
        mat = mat - mat_bar
        return mat.T @ mat

    def sample_factor_w(self, sparse_mat, tau_sparse_mat, tau_ind, W, X, tau, beta0=1, vargin=0):
        """Sampling N-by-R factor matrix W and its hyperparameters (mu_w, Lambda_w)."""

        dim1, rank = W.shape
        W_bar = np.mean(W, axis=0)
        temp = dim1 / (dim1 + beta0)
        var_W_hyper = inv(np.eye(rank) + self.cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
        var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_W_hyper)
        var_mu_hyper = self.mvnrnd_pre(temp * W_bar, (dim1 + beta0) * var_Lambda_hyper)

        if dim1 * rank ** 2 > 1e+8:
            vargin = 1

        if vargin == 0:
            var1 = X.T
            var2 = kr_prod(var1, var1)
            var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
            var4 = var1 @ tau_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
            for i in range(dim1):
                W[i, :] = self.mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
        elif vargin == 1:
            for i in range(dim1):
                pos0 = np.where(sparse_mat[i, :] != 0)
                Xt = X[pos0[0], :]
                var_mu = tau[i] * Xt.T @ sparse_mat[i, pos0[0]] + var_Lambda_hyper @ var_mu_hyper
                var_Lambda = tau[i] * Xt.T @ Xt + var_Lambda_hyper
                W[i, :] = self.mvnrnd_pre(solve(var_Lambda, var_mu), var_Lambda)

        return W

    def mnrnd(self, M, U, V):
        """
        Generate matrix normal distributed random matrix.
        M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
        """
        dim1, dim2 = M.shape
        X0 = np.random.randn(dim1, dim2)
        P = cholesky_lower(U)
        Q = cholesky_lower(V)

        return M + P @ X0 @ Q.T

    def sample_var_coefficient(self, X, time_lags):
        dim, rank = X.shape
        d = time_lags.shape[0]
        tmax = np.max(time_lags)

        Z_mat = X[tmax: dim, :]
        Q_mat = np.zeros((dim - tmax, rank * d))
        for k in range(d):
            Q_mat[:, k * rank: (k + 1) * rank] = X[tmax - time_lags[k]: dim - time_lags[k], :]
        var_Psi0 = np.eye(rank * d) + Q_mat.T @ Q_mat
        var_Psi = inv(var_Psi0)
        var_M = var_Psi @ Q_mat.T @ Z_mat
        var_S = np.eye(rank) + Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
        Sigma = invwishart.rvs(df=rank + dim - tmax, scale=var_S)

        return self.mnrnd(var_M, var_Psi, Sigma).astype('float32'), Sigma.astype('float32')

    def sample_factor_x(self, tau_sparse_mat, tau_ind, time_lags, W, X, A, Lambda_x):
        """Sampling T-by-R factor matrix X."""

        dim2, rank = X.shape
        tmax = np.max(time_lags)
        tmin = np.min(time_lags)
        d = time_lags.shape[0]
        A0 = np.dstack([A] * d)
        for k in range(d):
            A0[k * rank: (k + 1) * rank, :, k] = 0
        mat0 = Lambda_x @ A.T
        mat1 = np.einsum('kij, jt -> kit', A.reshape([d, rank, rank]), Lambda_x)
        mat2 = np.einsum('kit, kjt -> ij', mat1, A.reshape([d, rank, rank]))

        var1 = W.T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + Lambda_x[:, :, None]
        var4 = var1 @ tau_sparse_mat
        for t in range(dim2):
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
            index = list(range(0, d))
            if t >= dim2 - tmax and t < dim2 - tmin:
                index = list(np.where(t + time_lags < dim2))[0]
            elif t < tmax:
                Qt = np.zeros(rank)
                index = list(np.where(t + time_lags >= tmax))[0]
            if t < dim2 - tmin:
                Mt = mat2.copy()
                temp = np.zeros((rank * d, len(index)))
                n = 0
                for k in index:
                    temp[:, n] = X[t + time_lags[k] - time_lags, :].reshape(rank * d)
                    n += 1
                temp0 = X[t + time_lags[index], :].T - np.einsum('ijk, ik -> jk', A0[:, :, index], temp)
                Nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)

            var3[:, :, t] = var3[:, :, t] + Mt
            if t < tmax:
                var3[:, :, t] = var3[:, :, t] - Lambda_x + np.eye(rank)
            X[t, :] = self.mvnrnd_pre(solve(var3[:, :, t], var4[:, t] + Nt + Qt), var3[:, :, t])

        return X

    def sample_precision_tau(self, sparse_mat, mat_hat, ind):
        var_alpha = 1e-6 + 0.5 * np.sum(ind, axis=1)
        var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind, axis=1)
        return np.random.gamma(var_alpha, 1 / var_beta)

    def sample_precision_scalar_tau(self, sparse_mat, mat_hat, ind):
        var_alpha = 1e-6 + 0.5 * np.sum(ind)
        var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind)
        return np.random.gamma(var_alpha, 1 / var_beta)

    def execute(self, dense_mat, sparse_mat):
        """Bayesian Temporal Matrix Factorization, BTMF."""

        args = self.args
        dim1, dim2 = sparse_mat.shape
        d = np.array(args.time_lags).shape[0]

        time_lags = np.array(args.time_lags)
        W = np.random.randn(dim1, args.rank).astype('float32')
        X = np.random.randn(dim2, args.rank).astype('float32')

        if not np.isnan(sparse_mat).any():
            ind = sparse_mat != 0
            pos_test = np.where((dense_mat != 0) & (sparse_mat == 0))
        else:
            pos_test = np.where((dense_mat != 0) & (np.isnan(sparse_mat)))
            ind = ~np.isnan(sparse_mat)
            sparse_mat[np.isnan(sparse_mat)] = 0
        del dense_mat
        tau = np.ones(dim1)
        W_plus = np.zeros((dim1, args.rank)).astype('float32')
        X_plus = np.zeros((dim2, args.rank)).astype('float32')
        A_plus = np.zeros((args.rank * d, args.rank)).astype('float32')
        temp_hat = np.zeros(len(pos_test[0])).astype('float32')
        mat_hat_plus = np.zeros((dim1, dim2)).astype('float32')
        for it in trange(args.burn_iter + args.gibbs_iter):
            tau_ind = tau[:, None] * ind
            tau_sparse_mat = tau[:, None] * sparse_mat
            W = self.sample_factor_w(sparse_mat, tau_sparse_mat, tau_ind, W, X, tau)
            A, Sigma = self.sample_var_coefficient(X, time_lags)
            X = self.sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, inv(Sigma))
            mat_hat = W @ X.T
            if args.option == "factor":
                tau = self.sample_precision_tau(sparse_mat, mat_hat, ind)
            elif args.option == "pca":
                tau = self.sample_precision_scalar_tau(sparse_mat, mat_hat, ind)
                tau = tau * np.ones(dim1)
            temp_hat += mat_hat[pos_test]

            if it + 1 > args.burn_iter:
                W_plus += W
                X_plus += X
                A_plus += A
                mat_hat_plus += mat_hat
        mat_hat = mat_hat_plus / args.gibbs_iter

        return mat_hat


def get_matrix(args):
    matrix = None
    if args.dataset == 'taxi':
        matrix = np.load('dataset/taxi.npy')[:1464].astype('float32')

    if args.dataset == 'mat':
        matrix = np.load('dataset/mat.npy')[: 84839].astype('float32')
    return matrix


def GetSampling(matrix, ratio=0.05):
    rowIdx, colIdx = matrix.nonzero()
    p = np.random.permutation(len(rowIdx))
    rowIdx, colIdx = rowIdx[p], colIdx[p]
    sample = int(np.prod(matrix.shape) * ratio)
    rowIdx = rowIdx[:sample]
    colIdx = colIdx[:sample]
    sparseMat = np.zeros(matrix.shape)
    sparseMat[rowIdx, colIdx] = matrix[rowIdx, colIdx]

    mask = np.zeros(matrix.shape).astype('float32')
    mask[rowIdx, colIdx] = 1
    return sparseMat.astype('float32'), mask


def ErrMetrics(true, pred):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    ER = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return ER, NMAE


def run(runid, args):
    # Get Dataset
    matrix = get_matrix(args)
    global NR_min, NMAE_min

    matrix = np.load('./dataset/mat(2).npy', allow_pickle=True)

    denseMat = matrix.transpose()
    denseMat = denseMat.transpose()

    denseMat_backup = denseMat
    print(denseMat.shape)
    denseMat = denseMat[0: 15, :]

    thsh = np.percentile(denseMat, q=args.quantile)
    denseMat[denseMat > thsh] = thsh
    denseMat /= thsh

    # Sampling
    sparseMat, mask = GetSampling(denseMat, args.density)
    model = BTMF(args)
    recon = model.execute(dense_mat=denseMat, sparse_mat=sparseMat)

    recon_backup = recon

    testMask = 1 - mask


    if args.dataset == 'mat':
        testMask = testMask[:, 5952: 8928]
        denseMat = denseMat[:, 5952: 8928]
        recon = recon[:, 5952: 8928]  # 5952

    if args.dataset == 'ans_min':
        testMask = testMask[:, 60: 93]
        denseMat = denseMat[:, 60: 93]
        recon = recon[:, 60: 93]  # 33

    idx = np.nonzero(testMask)

    realVec = denseMat[idx]  # 真实值
    estiVec = recon[idx]     # 估计值
 
    recon = recon * thsh

    ER, NMAE = ErrMetrics(realVec * thsh, estiVec * thsh)

    recon = np.array(recon)
    print(recon)

    logger.info(f"Run ID={runid}, ER={ER:.3f}, NMAE={NMAE:.3f}")

    return ER, NMAE, recon_backup * thsh


def main(args):
    RunERs, RunNMAEs = [], []
    ans = np.array([])
    for runid in range(args.rounds):
        ER, NMAE, ans = run(runid, args)
        RunERs += [ER]
        RunNMAEs += [NMAE]

    print(RunERs, RunNMAEs)
    logger.info(f'Run ER={np.mean(RunERs):.3f} Run NMAE={np.mean(RunNMAEs):.3f}')

    data = pd.DataFrame(ans)
    data.to_csv('预测结果.csv', index=False)
    # data.to_csv('预测结果最小值.csv', index = False)
    # data.to_csv('预测结果最大值.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)  # 数据集
    parser.add_argument('--density', type=float)  # 精度设置
    parser.add_argument('--quantile', type=int, default=99)  # 设置为100，等同于归一化
    parser.add_argument('--rounds', type=int, default=5)  # 训练次数
    parser.add_argument('--burn_iter', type=int, default=30)  # 迭代次数
    parser.add_argument('--gibbs_iter', type=int, default=1)  # 一般设置为 10 - 20 （不需要管）
    parser.add_argument('--option', type=str, default='pca', choices=['pca', 'factor'])  # 主成分分析？？？？
    parser.add_argument('--time_lags', type=list, default=[1, 12])  # 一般设置为 1， 2， 12
    parser.add_argument('--rank', type=int, default=10)  # 矩阵的秩

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=f'results/BTMF_{args.dataset}_{args.density}.log', filemode='w')

    logger = logging.getLogger('BTMF')

    logger.info(f'Experiment Config = {args}')

    main(args)