from dolfin import *
import pylab
parameters['reorder_dofs_serial'] = False

r0 = 0.2
r1 = 1.
u0 = Constant(1)
u1 = Constant(1)
mesh = IntervalMesh(1000, r0, r1)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

def inner_boundary(x, on_boundary):
	return near(x[0], r0) and on_boundary

def outer_boundary(x, on_boundary):
	return near(x[0], r1) and on_boundary

bc0 = DirichletBC(V, u0, inner_boundary)
bc1 = DirichletBC(V, u1, outer_boundary)

r = Expression("x[0]")
F = inner(grad(v), grad(u)) * r * dx == Constant(0) * v * r * dx

u_ = Function(V)

pylab.figure(figsize=(8,4))
for u00, u11 in zip([1, 1, 1],[1, -1, 2]):
	u0.assign(u00)
	u1.assign(u11)
	solve(F, u_, bcs=[bc0, bc1])
	pylab.plot(u_.vector().array(), mesh.coordinates())

pylab.legend(["(a)", "(b)", "(c)"], loc="lower left")
pylab.xlabel("Velocity")
pylab.ylabel("r")
pylab.savefig("problem3-2.pdf")

#u_exact = (u1-u0)*ln(r)/(ln(r1/r0) + u0 - (u1-u0)*ln(r0)/ln(r1/r0)
#u_exact = project(u_exact,V)