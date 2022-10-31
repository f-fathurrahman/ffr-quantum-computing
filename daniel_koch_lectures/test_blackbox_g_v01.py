def f_v1(q):
    return q

# f(0,1) -> (1,0)
def f_v2(q):
    if q == 0:
        return 1
    elif q == 1:
        return 0        

def f_v3(q):
    return 0

def f_v4(q):
    return 1

def g(f, q1, q2):
    gq1 = q1
    gq2 = (q2 + f(q1)) % 2
    return gq1, gq2


for q1 in [0,1]:
    for q2 in [0,1]:
        print("%1d%1d -> %1d%1d" % (q1, q2, *g(f_v1, q1, q2)))


