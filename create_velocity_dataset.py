from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

#### UTILITY FUNCTIONS ####
def read_mesh_h5(mesh_path, mesh_name):

    hdf = HDF5File(MPI.comm_world, mesh_path+mesh_name+'.h5', 'r')
    mesh_h5 = Mesh()
    hdf.read(mesh_h5, '/mesh', False)

    markers = MeshFunction('size_t', mesh_h5, mesh_h5.topology().dim()) # Domain
    boundaries = MeshFunction('size_t', mesh_h5, mesh_h5.topology().dim() -1) 

    hdf.read(markers, '/markers')
    hdf.read(boundaries, '/boundaries')

    return mesh_h5, boundaries, markers


def write_hdf5(mesh_path, mesh_name):

    # Reads .xml file and returns .h5 file in postProcessing directory
    mesh_xml = Mesh(mesh_path + mesh_name + '.xml') 
    boundaries = MeshFunction('size_t',mesh_xml, File + "_facet_region.xml")
    markers = MeshFunction('size_t',meshObj, File + '_physical_region.xml')

    hdfw = HDF5File(mesh_xml.mpi_comm(), mesh_path + mesh_name +'.h5', "w")
    hdfw.write(mesh_xml, "/mesh")
    hdfw.write(markers, "/markers")
    hdfw.write(boundaries, '/boundaries')
    hdfw.close()







#========= Mesh data ==========
mesh_path = '/mnt/d/Meshes/'
mesh_name = 'Chip_Micronit_14e5Elem'

# Load mesh from xml file ~ mesh created with gmsh
# mesh = Mesh(mesh_path + mesh_name + '.xml')
# boundaries = MeshFunction('size_t', mesh_xml,mesh_path + mesh_name + '_facet_region.xml')
# markers = MeshFunction('size_t', mesh_xml,mesh_path + mesh_name + '_physical_region.xml')

# Load mesh from HDF5 file
mesh, boundaries, markers = read_mesh_h5(mesh_path, mesh_name)

# Print mesh dimension
tdim = mesh.topology().dim()
print('Mesh dimension: ', tdim)

# Print number of vertexes in mesh
nvert = len(mesh.coordinates())
print("Number of vertexes: ", nvert)

#=========== Load boundaries and subdomains ===============
n = FacetNormal(mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Get boundary markers from mesh (created with gmsh)
obstacleTags = [1]
inletTag = 2
outletTag = 3
Wall1Tag = 4
Wall2Tag = 5
fluidTag = 6

# Fluid Properties
rho = 1000
mu = 1
nu = Constant(0.01)

#====== Boundary conditions for inlet and outlet ===========
pInlet = 1000.0    # Pa
pOutlet = 0.0   # Pa

# Correction factor considering physical etching of the microfluidic device
corr_factor = 2e-5 #m ~ chip thickness correction factor/ Quasi-3D domain



#================== Function Spaces: Velocity and Pressure ====================
# Get Element Shape: Triangle, etc...
elementShape = mesh.ufl_cell()

Uel = VectorElement('Lagrange', elementShape, 2) # Velocity vector field
Pel = FiniteElement('Lagrange', elementShape, 1) # Pressure field
UPel = MixedElement([Uel,Pel])

# Mixed Function Space: Pressure and Velocity
W = FunctionSpace(mesh, UPel)

# Define test functions
(v,q) = TestFunctions(W)

# Define trial functions
w = Function(W)
(u,p) = (as_vector((w[0], w[1])), w[2])

#=================== Apply boundary conditions =================

bc = []
# No-slip condition for walls

for i in obstacleTags:
    bc0 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, i)
    bc.append(bc0)

bc.append(DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, Wall1Tag))
bc.append(DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, Wall2Tag))

# Inlet and outlet pressure conditions
bc.append(DirichletBC(W.sub(1), Constant(pInlet), boundaries, inletTag))
bc.append(DirichletBC(W.sub(1), Constant(pOutlet), boundaries, outletTag))


#====================== Variational form ==========================

# Linear Momentum Equation
    # Inertia Term            # Viscous Term                  # Pressure Term           # Continuity
F = inner(grad(u)*u, v)*dx() + nu*inner(grad(u), grad(v))*dx() - div(v)*p*dx() + q*div(u)*dx()

dw = TrialFunction(W)

# Calculate Jacobian Matrix
J = derivative(F,w,dw)

#================= Problem and Solver definitions =================
nsproblem = NonlinearVariationalProblem(F, w, bc, J)
solver = NonlinearVariationalSolver(nsproblem)

######## Configure numerical method
system_parameters = solver.parameters
system_parameters['nonlinear_solver'] = 'newton'
system_parameters['newton_solver']['absolute_tolerance'] =  1e-10
system_parameters['newton_solver']['relative_tolerance'] = 1e-7
system_parameters['newton_solver']['maximum_iterations'] = 500
system_parameters['newton_solver']['linear_solver'] = 'mumps'

#================= Solve problem in serial =================
# solver.solve()

#================= Solve problem in parallel =================
com = MPI.comm_world
rank = MPI.rank(com)
print(rank)

MPI.barrier(com)
solver.solve()

#================= Save results =================
u,p = w.leaf_node().split()

File("u.pvd") << u
File("p.pvd") << p


# plot(u)
# plt.show()


# x = mesh.coordinates()[:,0]
# y = mesh.coordinates()[:,1]
# n_vertex = len(x)
# shape = (n_vertex, 2) # 2D data

# u_sol = np.zeros(shape)

# # Store Pressure and Velocity valures in
# u__ = u.compute_vertex_values(mesh)
# p__ = p.compute_vertex_values(mesh)

