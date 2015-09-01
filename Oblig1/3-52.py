from dolfin import *
from math import log as ln, sinh, pi
set_log_active(False)

a=1.
b=0.25
c=0.5
mu = 0.2
dpdx = -0.1
F = (a**2 - b**2 + c**2)/(2*c)
M = sqrt(F**2 - a**2)
alpha = 0.5*ln((F + M)/(F - M))
beta = 0.5*ln((F - c + M)/(F - c - M))
s = 0
for i in range(1,20):
	s += (i*exp(-i*(beta + alpha)))/sinh(i*beta - i*alpha)

Qe=(pi/(8*mu)) * (-dpdx) * (a*a*a*a - b*b*b*b - (4*c*c*M*M)/(beta - alpha) - 8*c*c*M*M*s)

for i in range(1,5):
	E = []; h = [];
	for j in range(1,7):
		mesh = Mesh('eccentric%s.xml' % j)
		V = FunctionSpace(mesh, 'CG', i)
		u = TrialFunction(V)
		v = TestFunction(V)
		F = inner(grad(u), grad(v))*dx + 1/mu*dpdx*v*dx
		bc = DirichletBC(V, Constant(0), DomainBoundary())
		u_ = Function(V)
		solve(lhs(F) == rhs(F), u_, bcs=bc)
		Q = assemble(u_*dx)
		u_error = abs(Qe-Q)
		E.append(u_error)
		h.append(mesh.hmin())	

	print 'For degree %s:' % i
	for k in range(1, len(E)):
		r = ln(E[k]/E[k-1])/ln(h[k]/h[k-1])
		print 'h=%2.2E E=%2.2E r=%.2f' %(h[k], E[k], r)