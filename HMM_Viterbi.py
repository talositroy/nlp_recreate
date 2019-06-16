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


states = ('OUT1''OUT2', 'OUT3')
obs = ('INPUT1', 'INPUT2', 'INPUT3')
start_p = {'OUT1': 0.63, 'OUT2': 0.17, 'OUT3': 0.20}
trans_p = {
    'OUT1': {'OUT1': 0.50, 'OUT2': 0.375, 'OUT3': 0.125},
    'OUT2': {'OUT1': 0.25, 'OUT2': 0.125, 'OUT3': 0.625},
    'OUT3': {'OUT1': 0.25, 'OUT2': 0.375, 'OUT3': 0.375},
}
emit_p = {
    'OUT1': {'INPUT1': 0.60, 'INPUT2': 0.20, 'INPUT3': 0.15, 'INPUT4': 0.05},
    'OUT2': {'INPUT1': 0.25, 'INPUT2': 0.25, 'INPUT3': 0.25, 'INPUT4': 0.25},
    'OUT3': {'INPUT1': 0.05, 'INPUT2': 0.10, 'INPUT3': 0.35, 'INPUT4': 0.50},
}

print(hmm_viterbi(obs, states, start_p, trans_p, emit_p))
