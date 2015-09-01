from dolfin import *
from numpy import eye
set_log_active(False)

H = 0.41
D = 0.1
R = D/2.
Um = 0.3
nu = 0.001
rho = 1

mesh = Mesh('prob2.xml')
i = 4
for i in range(i):
	mesh = refine(mesh)

	n = -FacetNormal(mesh)
	n1 = as_vector((1.0,0))
	n2 = as_vector((0,1.0))
	nx = dot(n,n1)
	ny = dot(n,n2)
	nt = as_vector((ny,-nx))

	V = VectorFunctionSpace(mesh, 'CG', 2)
	Q = FunctionSpace(mesh, 'CG', 1)
	VQ = V * Q
	ug, pg = TrialFunctions(VQ)
	vg, qg = TestFunctions(VQ)
	u, p = TrialFunctions(VQ)
	v, q = TestFunctions(VQ)

	def top(x, on_boundary): return x[1] > (H-DOLFIN_EPS)
	def bottom(x, on_boundary): return x[1] < DOLFIN_EPS
	def circle(x, on_boundary): return sqrt((x[0]-0.2)**2 + (x[1]-0.2)**2) < (R		+DOLFIN_EPS)

	u_in = Expression(("4*Um*x[1]*(H-x[1])/(H*H)","0.0"),Um=Um,H=H)

	bc0 = DirichletBC(VQ.sub(0), Constant((0,0)), top)
	bc1 = DirichletBC(VQ.sub(0), Constant((0,0)), bottom)
	bc2 = DirichletBC(VQ.sub(0), Constant((0,0)), circle)
	inflow  = DirichletBC(VQ.sub(0), u_in, "x[0] < DOLFIN_EPS")
	bcs = [bc0, bc1, bc2, inflow]

	Fg = rho*nu*inner(grad(vg), grad(ug))*dx - div(vg)*pg*dx - q*div(ug)*dx + 			Constant(0)*qg*dx

	upg_ = Function(VQ)
	solve(lhs(Fg)==rhs(Fg), upg_, bcs=bcs)

	up_1 = Function(VQ)
	k = 0
	error = 1
	while k < 10 and error > 1e-5:

		F = rho*nu*inner(grad(u), grad(v))*dx + rho*inner(grad(u)*upg_.sub(0),v)*dx - div(v)*p*dx - q*div(u)*dx + Constant(0)*q*dx
		solve(lhs(F)==rhs(F), up_1, bcs=bcs)
		error = errornorm(upg_, up_1)
		upg_.assign(up_1)
		k += 1

	u_,p_ = upg_.split(True)


	Circle = AutoSubDomain(circle)
	mf = FacetFunction("size_t", mesh)
	mf.set_all(0)
	Circle.mark(mf, 1)
	ds = ds[mf]

	ut = dot(nt,u_)
	Uav = (2*Um)/3
	Fd = assemble((rho*nu*dot(grad(ut),n)*ny-p_*nx)*ds(1), exterior_facet_domains=mf, 	mesh=mesh)
	Fl = assemble(-(rho*nu*dot(grad(ut),n)*nx+p_*ny)*ds(1), exterior_facet_domains=mf, 	mesh=mesh)
	Cd = (2*Fd)/(rho*Uav**2*D)
	Cl = (2*Fl)/(rho*Uav**2*D)

	press = p_.compute_vertex_values()
	dp = press[5]-press[7]

	u,v = u_.split(True)
	uvals = u.compute_vertex_values()
	xmax = 0
	for j in range(len(uvals)):
		if uvals[j] < 0:
			if mesh.coordinates()[j][0] > xmax:
				xmax = mesh.coordinates()[j][0]
	La = xmax - 0.25
	print('Unknowns: %d Cd: %.4f Cl: %.4f, dp: %.4f La: %.4f' % (len(mesh.coordinates()), Cd, Cl, dp, La))
	
