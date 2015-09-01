from dolfin import *
import matplotlib.pyplot as plt
from numpy import zeros
set_log_active(False)

L = 6.
beta = -0.19898
mesh = IntervalMesh(10000, 0, L)
V = FunctionSpace(mesh, 'CG', 1)
VV = V * V
vf, vh = TestFunctions(VV)

bc0 = DirichletBC(VV, Constant((0,0)), "std::abs(x[0]) < 1e-10")
bc1 = DirichletBC(VV.sub(1), Constant(1), "std::abs(x[0]-%d) < 1e-10" % L)

xfun = Function(V)
xfun=interpolate(Expression("x[0]"),V)
xval = xfun.compute_vertex_values()
xdiff = xval[1]-xval[0]
plt.figure()
fh_=interpolate(Expression(("x[0]","1./L*x[0]"),L=L),VV)
f_,h_=split(fh_)
F = h_*vf*dx - f_.dx(0)*vf*dx - inner(grad(h_), grad(vh))*dx + f_*h_.dx(0)*vh*dx + 	beta*vh*dx - beta*h_**2*vh*dx
solve(F==0, fh_ , [bc0, bc1])

f_,h_= fh_.split(True)
	
hval = h_.compute_vertex_values()
hdx = project(h_.dx(0),V)
hdiff = hdx.compute_vertex_values()
print("beta = %f, f'' = %f" % (beta, hdiff[0]))
plt.plot(xval, hval, label='$ \\beta=%f $' % beta)
plt.legend(loc='center right')
plt.xlabel(r'$ \eta = y \sqrt{U(1+m)/2 \nu x} $')
plt.ylabel("f'")
plt.figure()

plt.plot(xval,hdiff, label='$ \\beta=%f $' % beta)
plt.legend(loc='center right')
plt.xlabel(r'$ \eta = y \sqrt{U(1+m)/2 \nu x} $')
plt.ylabel("f''")
plt.axis([0, 6, 0, 1.4])
plt.show()