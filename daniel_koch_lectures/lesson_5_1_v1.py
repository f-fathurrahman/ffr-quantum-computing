import numpy as np
import math

# function as first class citizen
# function is treated similar to variable

# functional programming
# Haskell, OCaml, Lisp, 

# Sebuah fungsi yang mengembalikan fungsi
def blackbox_f():

    # Constant functions
    F1 = lambda x : 0
    F2 = lambda x : 1

    # Balanced functions
    F3 = lambda x : x % 2
    F4 = lambda x : (x + 1)%2
    # Bisa juga menggunakan else if

    functions = [F1, F2, F3, F4]
    # generate index integer random: 0, 1, 2, 3
    idx_rand = np.random.randint(0, 4) # 4: exclusive

    print("idx_rand = ", idx_rand)

    f = functions[idx_rand]
    return f



F = blackbox_f()
print("f(0): ", F(0))
print("f(1): ", F(1))

if F(0) == F(1):
    print("Conclusion: f is constant")
else:
    print("Conclusion: f is balanced")

