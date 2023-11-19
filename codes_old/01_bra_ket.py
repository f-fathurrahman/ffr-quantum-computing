from sympy.physics.quantum import Bra, Ket
from sympy import symbols, I

ψ = Bra("ψ")
print(ψ)

print("For Bra psi")
print("hilbert_space = ", ψ.hilbert_space)
print("is_commutative = ", ψ.is_commutative)
print("dual = ", ψ.dual)
print("dual_class = ", ψ.dual_class())

ϕ = Ket("ϕ")
print("For Ket psi")
print("hilbert_space = ", ϕ.hilbert_space)
print("is_commutative = ", ϕ.is_commutative)
print("dual = ", ϕ.dual)
print("dual_class = ", ϕ.dual_class())


print("\nSome state")
state1 = (1/2 + I/2)*ϕ + (1/2)*ψ.dual 
print(state1)


print("\nSome kets with 0 and 1:")
k0 = Ket(0)
k1 = Ket(1)
state1 = 2*I*k0 - 4*k1
print(state1)


# Kets with compound labels
print("\nKet with compound labels")
n,m = symbols("n m")
k = Ket(n,m)
print(k)
