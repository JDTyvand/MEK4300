from dolfin import *
L = 4
mesh = IntervalMesh(50, 0, L)
V = FunctionSpace(mesh, 'CG', 1)
VV = V * V
vf, vh = TestFunctions(VV)

bc0 = DirichletBC(VV, Constant((0,0)), "std::abs(x[0]) < 1e-10")
bc1 = DirichletBC(VV.sub(1), Constant(1), "std::abs(x[0]-%d) < 1e-10" % L)

fh_=interpolate(Expression(("x[0]","x[0]")),VV)
f_,h_=split(fh_)

F = h_*vf*dx - f_.dx(0)*vf*dx - inner(grad(h_), grad(vh))*dx + 2*f_*h_.dx(0)*vh*dx + vh*dx - h_**2*vh*dx
solve(F==0, fh_ , [bc0, bc1])

f_,h_=split(fh_)

fplot = plot(f_)
fplot.write_png("148_Newton_F_plot")
hplot = plot(f_)
hplot.write_png("148_Newton_H_plot")