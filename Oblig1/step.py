from dolfin import *
from numpy import eye
set_log_active(False)

mu = 100.
mesh = Mesh('step.xml')
V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)
VQ = V * Q
u, p = TrialFunctions(VQ)
v, q = TestFunctions(VQ)

def top(x, on_boundary): return x[1] > (0.5-DOLFIN_EPS)

def bottom(x, on_boundary): return ((x[1]-0.1 < DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS) or x[1] < DOLFIN_EPS or abs(x[0]-0.5) < DOLFIN_EPS) and on_boundary

bc0 = DirichletBC(VQ.sub(0), Constant((0.0,0.0)), bottom)
bc1 = DirichletBC(VQ.sub(0), Constant((1.0,0.0)), top)

F = mu*inner(grad(v), grad(u))*dx - div(v)*p*dx - q*div(u)*dx + Constant(0)*q*dx

up_ = Function(VQ)
solve(lhs(F) == rhs(F), up_, [bc0, bc1])

u_,p_=up_.split()

u, v = u_.split()

w = v.dx(0) - u.dx(1)

psi = TrialFunction(Q)
psiv = TestFunction(Q)

n=FacetNormal(mesh)
psi_ = Function(Q)
grad_psi = as_vector((-v,u))
F = -inner(grad(psiv), grad(psi))*dx + psiv*dot(grad_psi,n)*ds + psiv*w*dx
solve(lhs(F) == rhs(F), psi_)
f = File('psi.pvd')
f<<psi_

def left(x, on_boundary): 
	return x[0] < 1e-12 and on_boundary
Left = AutoSubDomain(left)
mfl = FacetFunction("size_t", mesh)
mfl.set_all(0)
Left.mark(mfl, 1)

def right(x, on_boundary): 
	return x[0] > 1-1e-12 and on_boundary
Right = AutoSubDomain(right)
mfr = FacetFunction("size_t", mesh)
mfr.set_all(0)
Right.mark(mfr, 1)

Q_left = assemble(Constant(1)*u*ds(1), exterior_facet_domains=mfl, mesh=mesh)
Q_right = assemble(Constant(1)*u*ds(1), exterior_facet_domains=mfr, mesh=mesh)
print('The velocity flux into the domain is %f ' % Q_left)
print('The velocity flux out of the domain is %f ' %Q_right)

def boundary(x, on_boundary):
	return on_boundary

bc = DirichletBC(Q, 100., boundary)
bc.apply(psi_.vector())
min = psi_.vector().array().argmin()
x, y = mesh.coordinates()[min]
print('The location of the vortex is at x=%f, y=%f' % (x, y))

tau = -p_*Identity(2) + mu*(grad(u_) + transpose(grad(u_)))
Bottom = AutoSubDomain(bottom)
mfb = FacetFunction("size_t", mesh)
mfb.set_all(0)
Bottom.mark(mfb, 1)

stress = assemble(Constant(1)*dot(dot(tau,n),n)*ds(1), exterior_facet_domains=mfb, mesh=mesh)
print('The normal stress on the bottom wall is %f' % stress)


#plot(u_)
#plot(psi_)
#interactive()