text_string = r'研究生命的起源'
dic = ['研究', '研究生', '生命', '起源']
zx_result = []
nx_result = []


# 正向匹配
def zxfc(N):
    l_s = len(text_string)
    i = 0
    n = N
    while (i < l_s):
        tmp = text_string[i:i + n]
        if (tmp in dic or len(tmp) == 1):
            i += len(tmp)
            n = N
            zx_result.append(tmp)
        else:
            n -= 1
    print(zx_result)
    return zx_result


# 逆向匹配
def nxfc(N):
    i = len(text_string)
    n = N
    nx_result_tmp = []
    while (i > 0):
        if (i - n >= 0):
            tmp = text_string[i - n:i]
        else:
            tmp = text_string[0:i]
        if (tmp in dic or len(tmp) == 1):
            i -= len(tmp)
            n = N
            nx_result_tmp.append(tmp)
        else:
            n -= 1
    for i in range(len(nx_result_tmp) - 1, -1, -1):
        nx_result.append(nx_result_tmp[i])
    print(nx_result)
    return nx_result


nxfc(4)


# 双向匹配
def sxppfc(N):
    if (len(zxfc(N)) > len(nxfc(N))):
        print(nx_result)
    else:
        print(zx_result)
