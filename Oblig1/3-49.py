from dolfin import *
from math import log as ln
set_log_active(False)

a=2.
mu = 0.2
dpdx = -0.1

ue=Expression('(-dpdx)/(2*sqrt(3)*a*mu) * (x[1] - 0.5*a*sqrt(3)) * (3*x[0]*x[0] - x[1]*x[1])', dpdx=dpdx, mu=mu, a=a)

for i in range(1,6):
	E = []; h = [];
	for j in range(1,7):
		mesh = Mesh('triangle%s.xml' % j)
		V = FunctionSpace(mesh, 'CG', i)
		u = TrialFunction(V)
		v = TestFunction(V)
		F = inner(grad(u), grad(v))*dx + 1/mu*dpdx*v*dx
		bc = DirichletBC(V, Constant(0), DomainBoundary())
		u_ = Function(V)
		solve(lhs(F) == rhs(F), u_, bcs=bc)
		uex = project(ue, V)
		u_error = errornorm(uex, u_, degree_rise=0)
		E.append(u_error)
		h.append(mesh.hmin())
	
	print 'For degree %s:' % i
	for k in range(1, len(E)):
		r = ln(E[k]/E[k-1])/ln(h[k]/h[k-1])
		print 'h=%2.2E E=%2.2E r=%.2f' %(h[k], E[k], r)