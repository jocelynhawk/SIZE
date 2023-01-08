import numpy as np
from numpy import shape
import pyvista as pv
from cmath import *
import statistics as stat
import scipy
from scipy import interpolate
from scipy.optimize import curve_fit
import trimesh

def mesh_reconstruction(filename):
    mesh = pv.read(filename,file_format='ply')
    print("mesh: ",mesh)
    mesh.plot(show_edges=True)
    
    return mesh

def principal_axis(mesh0,mesh1,mesh2):

    #merge meshes together
    mesh01 = pv.PolyDataFilters.merge(mesh0,mesh1)
    total_mesh = pv.PolyDataFilters.merge(mesh01,mesh2).extract_surface()
    total_mesh.plot(show_edges=True)
    total_mesh=total_mesh.triangulate()

    #convert to trimesh and calculate prinipal axes, CoM, and transformation matrix
    faces_as_array = total_mesh.faces.reshape((total_mesh.n_faces, 4))[:, 1:] 
    tmesh = trimesh.Trimesh(total_mesh.points, faces_as_array)
    principal_axes = tmesh.principal_inertia_vectors
    principal_T = tmesh.principal_inertia_transform
    com = tmesh.center_mass

    #Plot mesh with principal axes
    pl = pv.Plotter()
    pl.add_mesh(total_mesh,style='wireframe')
    pl.add_arrows(cent=com,direction=principal_axes[0],mag=20)
    pl.add_arrows(cent=com,direction=principal_axes[1],mag=20)
    pl.add_arrows(cent=com,direction=principal_axes[2],mag=20)
    pl.show()
    print("axes: ",principal_axes)
    print("PT: ",principal_T)

    return principal_T

def perform_calculations(PT,mesh):

    #align mesh with principal axes
    aligned_mesh = mesh.transform(PT,inplace=False).triangulate().extract_surface().subdivide(nsub=2)

    #plot original and rotated mesh
    pl = pv.Plotter()
    pl.add_mesh(mesh,style='wireframe', color='b')
    pl.add_mesh(aligned_mesh,style='wireframe', color='r')
    pl.add_axes_at_origin()
    #pl.show()

    point_data = np.round(aligned_mesh.points,decimals=1)
    xy_coords = np.unique(point_data[:,:2],axis=0)
    #print("xc: ",xy_coords, xy_coords.shape,point_data.shape)

    lengths=[]
    for xy in xy_coords:
        points=[]
        indices = np.where((point_data[:,0] == xy[0]) & (point_data[:,1]==xy[1]))
        #print("indices: ",indices)
        for i in indices:
            points.append(point_data[i,2])
        #print("points: ",points)
        #print("max l: ",np.max(points))
        #print("min l: ",np.min(points))
        lengths.append(np.max(points)-np.min(points))
 
    #print("lengths: ",lengths)
    length=max(lengths)
    volume=mesh.volume
    CSA = volume/length
    #print("i: ",indices)
    #print("length: ", length)
    #print("volume: ",mesh.volume)

    return [length,volume,CSA]
    


subject_path = 'C:\\Users\\jocel\\OneDrive\\Desktop\\20220607 Formal Data Collection\\Sub00000009\\Trial 1\\'
#'C:\\Users\\jocel\\Box\\lihandlab\\2022\\20222222 Project Hawk SIZE\\20220504 Data Collection\\Sub00000001'
APB_filename = subject_path + 'APB.ply'
FPBSH_filename = subject_path + 'FPB.ply'
FPBDH_filename = subject_path + 'FPBDH.ply'
OPP_filename = subject_path + 'OPP.ply'

APB, FPBSH, OPP, = mesh_reconstruction(APB_filename), mesh_reconstruction(FPBSH_filename), mesh_reconstruction(OPP_filename)
try:
    FPBDH = mesh_reconstruction(FPBDH_filename)
    FPB = pv.PolyDataFilters.merge(FPBSH,FPBDH,)
except:
    FPB = FPBSH
PT = principal_axis(APB,FPB,OPP)
APB_results, FPB_results, OPP_results = perform_calculations(PT,APB), perform_calculations(PT,FPB), perform_calculations(PT,OPP)

result_strings = ['length: ', 'volume: ', 'CSA: ']

print("APB Results: ")
for i in range(0,3):
    print(result_strings[i],APB_results[i])
print("FPB Results: ")
for i in range(0,3):
    print(result_strings[i],FPB_results[i])
print("OPP Results: ")
for i in range(0,3):
    print(result_strings[i],OPP_results[i])
