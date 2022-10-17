import numpy as np
import math

def blackbox_f():

    def F1(x):
        return 0

    def F2(x):
        return 1

    def F3(x):
        return x%2

    def F4(x):
        return (x+1)%2

    functions = [F1, F2, F3, F4]
    idx_rand = np.random.randint(0, 4) # 4: exclusive
    f = functions[idx_rand]
    return f

F = blackbox_f()
print("f(0): ", F(0))
print("f(1): ", F(1))

if F(0) == F(1):
    print("Conclusion: f is constant")
else:
    print("Conclusion: f is balanced")
