# @Time : 2022-03-02 16:02
# @Author : Phalange
# @File : HMM.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import numpy as np

class HiddenMarkov:
    def __init__(self):
        self.alphas = None
        self.forward_P = None
        self.backward_P = None

    # 前向算法
    def forward(self,Q,V,A,B,O,PI):
        # 状态序列的大小
        N = len(Q)
        M = len(O)
        # 初始化前向概率alpha值
        alphas = np.zeros((N,M))
        # 时刻数=观测序列数
        T = M
        # 遍历每一个时刻，计算前向概率alpha值
        for t in range(T):
            # 得到序列对应的索引
            indexOFO = V.index(O[t])
            # 遍历状态序列
            for i in range(N):
                if t == 0:
                    # P176 公式（10.15
                    alphas[i][t] = PI[t][i] * B[i][indexOFO]
                    print('alpha1(%d) = p%db%db(o1) = %f' %
                          (i+1,i,i,alphas[i][t]))
                else:
                    # 176 公式10.16
                    alphas[i][t] = np.dot([alpha[t - 1] for alpha in alphas],
                                          [a[i] for a in A]) * B[i][indexOFO]
                    print('alpha%d(%d) = [sigma alpha%d(i)ai%d]b%d(o%d) = %f' %
                          (t + 1, i + 1, t - 1, i, i, t, alphas[i][t]))

        # P176公式(10.17)
        self.forward_P = np.sum([alphas[M-1] for alpha in alphas])
        self.alphas = alphas

    # 后向算法
    def backward(self,Q,V,A,B,O,PI):
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        betas = np.ones((N,M))

        for i in range(N):
            print('beta%d(%d) = 1' % (M,i+1))
        # 对观测序列逆向遍历
        for t in range(M - 2,-1,-1):
            # 得到序列对应的索引值
            indexOfO = V.index(O[t + 1])
            # 遍历状态序列
            for i in range(N):
                # P178公式10.20
                betas[i][t] = np.dot(
                    np.multiply(A[i],[b[indexOfO] for b in B]),[beta[t+1] for beta in betas])
                realT = t + 1
                realI = i + 1
                print('beta%d(%d) = sigma[a%djbj(o%d)beta%d(j)] = (' %
                      (realT, realI, realI, realT + 1, realT + 1),
                      end='')
                for j in range(N):
                    print("%.2f * %.2f * %.2f + " %
                          (A[i][j], B[j][indexOfO], betas[j][t + 1]),
                          end='')
                print("0) = %.3f" % betas[i][t])
        # 取出第一个值
        indexOfO = V.index(O[0])
        self.betas = self.betas
        # P178公式 10.21
        P = np.dot(np.multiply(PI,[b[indexOfO] for b in B]),
                   [beta[0] for beta in betas])
        self.backward_P = P
        print("P(O|lambda) = ", end="")
        for i in range(N):
            print("%.1f * %.1f * %.5f + " %
                  (PI[0][i], B[i][indexOfO], betas[i][0]),
                  end="")
        print("0 = %f" % P)

    # 维特比算法
    def viterbi(self,Q,V,A,B,O,PI):
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # 初始化daltas
        deltas = np.zeros((N,M))
        # 初始化psis
        psis = np.zeros(N,M)
        # 初始化最优路径矩阵，该矩阵维度与观测序列维度相同
        I = np.zeros((1,M))
        # 遍历观测序列
        for t in range(M):
            # 从t=2开始
            realT = t +1
            # 得到序列对应的索引
            indexOfO = V.index(O[t])
            for i in range(N):
                realI = i+1
                if t == 0:
                    # P185 算法10.5 步骤1
                    deltas[i][t] = PI[0][i] * B[i][indexOfO]
                    psis[i][t] = 0
                    print('delta1(%d) = pi%d * b%d(o1) = %.2f * %.2f = %.2f' %
                          (realI, realI, realI, PI[0][i], B[i][indexOfO],
                           deltas[i][t]))
                    print('psis1(%d) = 0' % (realI))
                else:
                    # P185 算法10.5 步骤(2)
                    deltas[i][t] = np.max(
                        np.multiply([delta[t - 1] for delta in deltas],
                                    [a[i] for a in A]) * B[i][indexOfO])
                    print(
                        'delta%d(%d) = max[delta%d(j)aj%d]b%d(o%d) = %.2f * %.2f = %.5f'
                        % (realT, realI, realT - 1, realI, realI, realT,
                           np.max(
                               np.multiply([delta[t - 1] for delta in deltas],
                                           [a[i] for a in A])), B[i][indexOfO],
                           deltas[i][t]))
                    psis[i][t] = np.argmax(
                        np.multiply([delta[t - 1] for delta in deltas],
                                    [a[i] for a in A]))
                    print('psis%d(%d) = argmax[delta%d(j)aj%d] = %d' %
                            (realT, realI, realT - 1, realI, psis[i][t]))
        # print(deltas)
        # print(psis)
        # 得到最优路径的终结点
        I[0][M - 1] = np.argmax([delta[M - 1] for delta in deltas])
        print('i%d = argmax[deltaT(i)] = %d' % (M, I[0][M - 1] + 1))
        # 递归由后向前得到其他结点
        for t in range(M - 2, -1, -1):
            I[0][t] = psis[int(I[0][t + 1])][t + 1]
            print('i%d = psis%d(i%d) = %d' %
                  (t + 1, t + 2, t + 2, I[0][t] + 1))
        # 输出最优路径
        print('最优路径是：', "->".join([str(int(i + 1)) for i in I[0]]))