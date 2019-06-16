import numpy as np


def hmm_viterbi(obs, states, start_p, trans_p, emit_p):
    # obs观察序列
    # states隐状态
    # start_p初始概率
    # trans_p状态转移矩阵
    # emit_p发射概率（混淆矩阵）
    # 假设观察序列数为X，序列种类为N，状态数为M,则obs长度为X，states长度为X，start_p长度为M，trans_p形状为MxM,emit_p形状为MxN
    V = [{}]
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
    for t in range[1, len(obs)]:
        V.append({})
        for y in states:
            V[t][y] = max([(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]]) for y0 in states])
            result = []
    for vector in V:
        temp = {}
        temp[vector.keys()[np.argmax(vector.values())]] = max(vector.values())
        result.append(temp)
    return result
