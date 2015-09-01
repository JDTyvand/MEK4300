from dolfin import *
h = 1.
N = 10
mesh = IntervalMesh(N, -h, h)
V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = Testfunction(V)

def bottom(x, on_boundary):
	return near(x[0], -h) and on_boundary

def top(x, on_boundary):
	return near(x[0], h) and on_boundary

U = 1.
bcs = [DirichletBC(V, 0, bottom),
       DirichletBD(V, U, top)]

u_ = Function(V)
solve(-inner(grad(u, grad(v))*dx == Constant(0)*v*dx, u_, bcs=bcs)

u_exact = project(Expression("U/2*(1+x[0]/h)", U=U, h=h), V)

plot(u_, title="Couette flow")
hold(on)
plot(u_, -u_exact, title="Error Couette flow")