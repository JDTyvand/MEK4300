from dolfin import *
from numpy import zeros,diff
import matplotlib.pyplot as plt
set_log_active(False)

L = 6.
Beta = [1.,0.3,0,-0.1,-0.18,-0.198838]
mesh = IntervalMesh(10000, 0, L)
V = FunctionSpace(mesh, 'CG', 1)
VV = V * V
vf, vh = TestFunctions(VV)

bc0 = DirichletBC(VV, Constant((0,0)), "std::abs(x[0]) < 1e-12")
bc1 = DirichletBC(VV.sub(1), Constant(1), "std::abs(x[0]-%d) < 1e-12" % L)

xfunc = Function(V)
xfunc = interpolate(Expression("x[0]"),V)
xvals = xfunc.compute_vertex_values()
xdiff = xvals[1]-xvals[0]
hdx = []

plt.figure()
for beta in Beta:
	fh_=interpolate(Expression(("x[0]","1./L*x[0]"),L=L),VV)
	f_,h_=split(fh_)
	F = h_*vf*dx - f_.dx(0)*vf*dx - inner(grad(h_), grad(vh))*dx + f_*h_.dx(0)*vh*dx + 	beta*vh*dx - beta*h_**2*vh*dx
	solve(F==0, fh_ , [bc0, bc1])

	f_,h_= fh_.split(True)
	
	hvals = h_.compute_vertex_values()
	
	hdiff = project(h_.dx(0),V)
	hdx.append(hdiff.compute_vertex_values())

	plt.plot(xvals, hvals, label='$ \\beta=%f $' % beta)
	plt.legend(loc='center right')

plt.xlabel(r'$ \eta = y \sqrt{U(1+m)/2 \nu x} $')
plt.ylabel("f'")
plt.axis([0, 6, 0, 1.2])

plt.figure()
for i in range(len(hdx)):
	plt.plot(xvals,hdx[i], label='$ \\beta=%f $' % Beta[i])
	plt.legend(loc='center right')

plt.xlabel(r'$ \eta = y \sqrt{U(1+m)/2 \nu x} $')
plt.ylabel("f''")
plt.axis([0, 6, 0, 1.4])
plt.show()