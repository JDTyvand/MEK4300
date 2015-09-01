from dolfin import *

mu = 100.
mesh = UnitSquareMesh(30, 30)
V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)
VQ = V * Q
u, p = TrialFunctions(VQ)
v, q = TestFunctions(VQ)

def top(x, on_boundary): return x[1] > (1-DOLFIN_EPS)

bc0 = DirichletBC(VQ.sub(0), Constant((0.0,0.0)), DomainBoundary())
bc1 = DirichletBC(VQ.sub(0), Constant((1.0,0.0)), top)

F = mu*inner(grad(v), grad(u))*dx - div(v)*p*dx - q*div(u)*dx + Constant(0)*q*dx

up_ = Function(VQ)
solve(lhs(F) == rhs(F), up_, [bc0, bc1])

u_,p_=up_.split()

u, v = u_.split()

w = v.dx(0) - u.dx(1)

psi = TrialFunction(Q)
psiv = TestFunction(Q)

bcs = DirichletBC(Q, Constant(0), DomainBoundary())

psi_ = Function(Q)
solve(-inner(grad(psiv), grad(psi))*dx == -psiv*w*dx, psi_, bcs=bcs)

min = psi_.vector().array().argmin()
x, y = mesh.coordinates()[min]
print('The location of the vortex is at x=%f, y=%f' % (x, y))

plot(u_)
plot(psi_)
interactive()