a = 0.04
wf = 9681
qf = 40.1e3
q = 4250
w = a * wf * qf / q
print(w)
aa = pow(w / 1000, 1 / 3)
print(aa)
p = 0.1
r = pow((83.641 * pow(aa, 0.087) / p), 1 / 1.087)
print(r)

# xp = 83.641 * pow((20), -2.087)
# print(xp)

print(83.641 * pow(1.54, 0.087) / pow(505, 1.087))
