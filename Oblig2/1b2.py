from dolfin import *
import matplotlib.pyplot as plt
from numpy import zeros
set_log_active(False)

L = 6.
beta = -0.1
mesh = IntervalMesh(10000, 0, L)
V = FunctionSpace(mesh, 'CG', 1)
VV = V * V


bc0 = DirichletBC(VV, Constant((0,0)), "std::abs(x[0]) < 1e-10")
bc1 = DirichletBC(VV.sub(1), Constant(1), "std::abs(x[0]-%d) < 1e-10" % L)

plt.figure()
pre = [1.0, -1.0]
solution=['Original solution', 'Second solution']

for i in range(2):
	
	vf, vh = TestFunctions(VV)
	xfun = Function(V)
	xfun=interpolate(Expression("x[0]"),V)
	xval = xfun.compute_vertex_values()
	xdiff = xval[1]-xval[0]
	fh_=interpolate(Expression(("pre*x[0]","pre*1./L*x[0]"),L=L, pre = pre[i]),VV)
	f_,h_=split(fh_)
	F = h_*vf*dx - f_.dx(0)*vf*dx - inner(grad(h_), grad(vh))*dx + f_*h_.dx(0)*vh*dx + 	beta*vh*dx - beta*h_**2*vh*dx
	solve(F==0, fh_ , [bc0, bc1])

	f_,h_= fh_.split(True)
	
	hval = h_.compute_vertex_values()
	hdx = project(h_.dx(0),V)
	hdiff = hdx.compute_vertex_values()
	plt.plot(xval,hdiff, label='%s' % solution[i])
	plt.title('The two solutions for $ \\beta $ = -0.1')
	plt.legend(loc='center right')
	plt.xlabel(r'$ \eta = y \sqrt{U(1+m)/2 \nu x} $')
	plt.ylabel("f''")
	
plt.show()