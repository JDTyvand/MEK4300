from dolfin import *
from numpy import zeros, savetxt, column_stack
import matplotlib.pyplot as plt
import sys
set_log_active(False)

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
mesh = Mesh("prob2.xml")
i = 2
for i in range(i):
	mesh = refine(mesh)
print len(mesh.coordinates())

n = -FacetNormal(mesh)
n1 = as_vector((1.0,0))
n2 = as_vector((0,1.0))
nx = dot(n,n1)
ny = dot(n,n2)
nt = as_vector((ny,-nx))

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Set parameter values
dt = 0.0005
T = 0.331
nu = 0.001
rho = 1
Um = 1.5
Uav = (2*Um)/3
H = 0.41
D = 0.1
R = D/2.
St = (D*(1/T))/Uav

# Define velocity boundary condition
u_in = Expression(("4*Um*x[1]*(H-x[1])/(H*H)","0.0"),Um=Um,H=H)

# Define boundary conditions
def top(x, on_boundary): return x[1] > (H-DOLFIN_EPS)
def bottom(x, on_boundary): return x[1] < DOLFIN_EPS
def circle(x, on_boundary): return sqrt((x[0]-0.2)**2 + (x[1]-0.2)**2) < (R+DOLFIN_EPS)

bc0 = DirichletBC(V, Constant((0,0)), top)
bc1 = DirichletBC(V, Constant((0,0)), bottom)
bc2 = DirichletBC(V, Constant((0,0)), circle)

inflow  = DirichletBC(V, u_in, "x[0] < DOLFIN_EPS")
outflow = DirichletBC(Q, 0, "x[0] > 2.2 - DOLFIN_EPS")
bcu = [bc0, bc1, bc2, inflow]
bcp = [outflow]

# Create functions
u0 = Function(V, 'u0.xml')
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Create files for storing solution
ufile = File("results/velocity.pvd")

CD = zeros(T/dt)
CL = zeros(T/dt)
DP = zeros(T/dt)
times = zeros(T/dt)

# Time-stepping
t = dt
counter = 0
loc = 0
while t < T + DOLFIN_EPS:
	print t
	times[loc] = t
 
	# Compute tentative velocity step
	begin("Computing tentative velocity")
	b1 = assemble(L1)
	[bc.apply(A1, b1) for bc in bcu]
	solve(A1, u1.vector(), b1, "gmres", "default")
	end()

	# Pressure correction
	begin("Computing pressure correction")
	b2 = assemble(L2)
	[bc.apply(A2, b2) for bc in bcp]
	solve(A2, p1.vector(), b2, "gmres", prec)
	end()

	# Velocity correction
	begin("Computing velocity correction")
	b3 = assemble(L3)
	[bc.apply(A3, b3) for bc in bcu]
	solve(A3, u1.vector(), b3, "gmres", "default")
	end()

	Circle = AutoSubDomain(circle)  
	mf = FacetFunction("size_t", mesh)
   	mf.set_all(0)
	Circle.mark(mf, 1)
    	ds = ds[mf]

	ut = dot(nt,u1)
	Fd = assemble((rho*nu*dot(grad(ut),n)*ny-p1*nx)*ds(1), exterior_facet_domains=mf, 	mesh=mesh)
	Fl = assemble(-(rho*nu*dot(grad(ut),n)*nx+p1*ny)*ds(1), exterior_facet_domains=mf, 	mesh=mesh)
	Cd = (2*Fd)/(rho*Uav**2*D)
	CD[loc] = Cd
	Cl = (2*Fl)/(rho*Uav**2*D)
	CL[loc] = Cl
	press = p1.compute_vertex_values()
	dp = press[5]-press[7]
	DP[loc] = dp
	"""
	# Save to file
	if counter % 30 == 0:
		ufile << u1
		counter += 1
	else:
		counter += 1
	"""
	# Move to next time step
	u0.assign(u1)
	t += dt
	loc += 1

#u0file = File('u0.xml')
#u0file << u0

output = column_stack((times.flatten(),CD.flatten()))
savetxt('CD.csv',output,fmt='%.5f',delimiter='	')
output = column_stack((times.flatten(),CL.flatten()))
savetxt('CL.csv',output,fmt='%.5f',delimiter='	')
output = column_stack((times.flatten(),DP.flatten()))
savetxt('DP.csv',output,fmt='%.5f',delimiter='	')

print('Maximum drag coefficient: %.4f' % max(CD))
print('Maximum lift coefficient: %.4f' % max(CL))
print('Strouhal number: %.4f' % St)
print('Pressure difference at midpoint: %.4f' % DP[331])

plt.figure()
plt.plot(times, CD)
plt.xlabel('t [s]')
plt.ylabel('$ C_D $')
plt.axis([0, times[-1], min(CD), max(CD)])
plt.figure()
plt.plot(times, CL)
plt.xlabel('t [s]')
plt.ylabel('$ C_L $')
plt.axis([0, times[-1], min(CL), max(CL)])
plt.figure()
plt.plot(times, DP)
plt.xlabel('t [s]')
plt.ylabel('$ \Delta p $')
plt.axis([0, times[-1], min(DP), max(DP)])
plt.show()
