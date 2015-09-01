from dolfin import *
from numpy import arctan, zeros
import matplotlib.pyplot as plt
set_log_active(False)

h = 1.
vs = 0.05
Re = 1000.
nu = vs/Re
k = 0.41
A = 26.

mesh = IntervalMesh(230, 0, h)
V = FunctionSpace(mesh, 'CG', 1)
ug = TrialFunction(V)
v = TestFunction(V)

bcs = DirichletBC(V, Constant(0), "std::abs(x[0]) < 1e-12")

x = mesh.coordinates()
xdiff = x[1]-x[0]
x[:,0] = h - (arctan(pi*(x[:,0])) / arctan(pi))
yplus = x*Re

l = Expression("k*x[0]*(1-exp(-x[0]*vs/(nu*A)))", k=k, vs=vs, A=A, nu=nu)

Fg = nu*inner(grad(v), grad(ug))*dx - (vs**2/h)*v*dx


u_ = Function(V)
solve(lhs(Fg)==rhs(Fg), u_, bcs = [bcs])

F = nu*v.dx(0)*u_.dx(0)*dx - (vs**2/h)*v*dx + l**2*abs(u_.dx(0))*u_.dx(0)*v.dx(0)*dx

solve(F==0, u_, bcs = [bcs])

uval = u_.compute_vertex_values()
uplus = uval/vs

plt.figure()
plt.plot(yplus,uplus)
plt.xlabel('$ y^+ $')
plt.ylabel('$ u^+ $')
plt.axis([0,5,0,5])

B = 5.5
u30 = interpolate(Expression("(1/k)*log(x[0]*Re) + B", k=k, B=B, Re=Re),V)
u30vals = u30.compute_vertex_values()
plt.figure()
plt.plot(yplus,uplus, label = 'calculated solution')
plt.plot(yplus,u30vals, label = 'theoretical solution')
plt.legend(loc='center right')
plt.xlabel('$ y^+ $')
plt.ylabel('$ u^+ $')
plt.axis([30,1000, 0, 25])

B = 5
u30 = interpolate(Expression("(1/k)*log(x[0]*Re) + B", k=k, B=B, Re=Re),V)
u30vals = u30.compute_vertex_values()
plt.figure()
plt.plot(yplus,uplus, label = 'calculated solution')
plt.plot(yplus,u30vals, label = 'theoretical solution')
plt.legend(loc='center right')
plt.xlabel('$ y^+ $')
plt.ylabel('$ u^+ $')
plt.axis([30,1000, 0, 25])

ll = interpolate(l,V)
lvals = ll.compute_vertex_values()
du = zeros(len(uval))

du = project(u_.dx(0),V)
print(du(1)*lvals[0])
plt.show()
