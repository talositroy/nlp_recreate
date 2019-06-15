text_string = r'研究生命的起源'
dic = ['研究', '研究生', '生命', '起源']


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
            print(tmp)
        else:
            n -= 1


# 逆向匹配
def nxfc(N):
    i = len(text_string)
    n = N
    while (i > 0):
        if (i - n >= 0):
            tmp = text_string[i - n:i]
        else:
            tmp = text_string[0:i]
        if (tmp in dic or len(tmp) == 1):
            i -= len(tmp)
            n = N
            print(tmp)
        else:
            n -= 1


nxfc(3)