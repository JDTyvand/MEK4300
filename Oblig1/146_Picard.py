from dolfin import *
L = 4
mesh = IntervalMesh(50, 0, L)
V = FunctionSpace(mesh, 'CG', 1)
VV = V * V
fh = TrialFunction(VV)
f, h = split(fh)
vf, vh = TestFunctions(VV)

bc0 = DirichletBC(VV, Constant((0,0)), "std::abs(x[0]) < 1e-10")
bc1 = DirichletBC(VV.sub(1), Constant(1), "std::abs(x[0]-%d) < 1e-10" % L)

fh_=interpolate(Expression(("x[0]","x[0]")),VV)
f_,h_=split(fh_)

F = h*vf*dx - f.dx(0)*vf*dx - inner(grad(h), grad(vh))*dx + f_*h.dx(0)*vh*dx + vh*dx - h*h_*vh*dx

k = 0
fh_1 = Function(VV)
while k < 100 and error > 1e-12:
	solve(lhs(F) == rhs(F), fh_, [bc0, bc1])
	error = errornorm(fh_, fh_1)
	fh_1.assign(fh_)
	print "Error = ", k, error
	k += 1

f_,h_=split(fh_)

plot(f_)
plot(h_)
interactive()