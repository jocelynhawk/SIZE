import numpy as np
from numpy import shape
import pyvista as pv
from cmath import *
import statistics as stat
import scipy
from scipy import interpolate
from scipy.optimize import curve_fit

def interpolate_points(point_data,surface_points):
    point_data=np.append(point_data,surface_points,axis=0)
    point_cloud = pv.PolyData(point_data)

    #create structured grid
    box = point_cloud.outline()
    bounds = box.bounds

    x=point_data[:,0]
    y=point_data[:,1]
    z=point_data[:,2]
    print("xyz: ",x,y,z)
    f= interpolate.interp2d(x,y,z, kind='linear',fill_value=0)
    xi=np.arange(bounds[0],bounds[1],0.5)
    yi=np.arange(bounds[2],bounds[3],0.5)
    xx,yy=np.meshgrid(xi,yi)
    xx=xx.flatten()
    yy=yy.flatten()
    zz=f(xi,yi).flatten()
    print("interp shapes: ",xx.shape,yy.shape,zz.shape)
    #interp_points = np.stack([xx,yy,zz],axis=1)
    first=True
    for i in range(0,zz.size):
        if zz[i]<bounds[5] and zz[i]>bounds[4]:
            if first==True:
                interp_points = np.array([xx[i],yy[i],zz[i]],ndmin=2)
                first=False
            else:
                interp_points=np.vstack([interp_points,np.array([xx[i],yy[i],zz[i]],ndmin=2)])
    print("interp pts: ",interp_points,interp_points.shape)
    interp_cloud=pv.PolyData(interp_points)
    pl=pv.Plotter()
    pl.add_mesh(box)
    pl.add_mesh(interp_cloud)
    pl.show()

    return interp_points 

def function(data, a0, a1,a2, a3, a4, a5,a6,a7,a8,a9):
    x = data[0]
    y = data[1]
    
    return a0+a1*x+a2*y+a3*x*y+a4*x**2+a5*y**2+a6*x*y**2+a7*x**2*y+a8*x**3+a9*y**3

def fit_surface(point_data,empty_cells, extra_xy):
    if side == 1:
        bounds = ((-np.inf,0,0,0,0.001,0.001),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,))
    else:
        bounds = ((-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,),(np.inf,0,0,0,-0.001,-0.001))
    x_data = []
    y_data = []
    z_data = []
    for item in point_data:
        x_data.append(item[0])
        y_data.append(item[1])
        z_data.append(item[2])
    # get fit parameters from scipy curve fit
    parameters0, covariance = curve_fit(function, [x_data, y_data], z_data)
    parameters1, covariance = curve_fit(function, [x_data, y_data], z_data)
    print("parameters: ",parameters1)


    # create coordinate arrays for vectorized evaluations
    #X, Y = np.meshgrid(model_x_data, model_y_data)
    X = empty_cells[:,0]
    Y = empty_cells[:,1]
    # calculate Z coordinate array
    Z = function(np.array([X, Y]), *parameters0)
    print("Z: ",Z)
    x,y,z =X.flatten(), Y.flatten(), Z.flatten()
    points0 = np.stack([x,y,z],axis=1)

    X = extra_xy[:,0]
    Y = extra_xy[:,1]
    # calculate Z coordinate array
    Z = function(np.array([X, Y]), *parameters1)
    print("Z: ",Z)
    x,y,z =X.flatten(), Y.flatten(), Z.flatten()
    points1 = np.stack([x,y,z],axis=1)
    points=np.vstack([points0,points1])
    point_cloud = pv.PolyData(points)   

    
    return points0

def remove_outliers(point_data,original_points):
    distances=[]
    points_checked=[]
    i=0
    point_data_copy=point_data
    std = stat.pstdev(original_points[:,2])
    med = stat.median(original_points[:,2])
    
    """for point0 in point_data:
        #print("point: ",i)
        points_checked.append(point0)
        min_distance=float(100000)
        #print("point0 loc: ", np.where(point_data_copy[0:2,:]==point0))
        #print("size of copy: ",point_data_copy)
        point_data_copy = np.delete(point_data_copy,0,0)
        print("points left: ",point_data_copy.shape[0])
        for point1 in point_data_copy:
            #print("points checked: ",points_checked)
            for checked_point in points_checked:
                if all(checked_point == point1):
                    #print("break:",checked_point,point1)
                    break
            distance=np.linalg.norm(point0-point1)
            if distance < min_distance:
                min_distance=distance
            #print("distance:", distance)
        distances.append(min_distance)
        i=i+1
    #print(distances)
    std = stat.pstdev(distances)
    med = stat.median(distances)"""
    
    
    print("med,std: ",med,std)
    for point in point_data:
        if abs(point[2]-med)<std*5:
            try:
                simplified_points=np.vstack([simplified_points,point])
            except:
                simplified_points = np.array([point])
        else:
            print('removed ',point)

    return simplified_points

def simplify_points(filename):
    point_data = np.loadtxt(filename, dtype=float)
    number_of_rows =point_data.shape[0]
    if number_of_rows>5000:  
        random_indices = np.random.choice(number_of_rows, size=5000, replace=False)
        point_data=point_data[random_indices,:]
    i=0

    point_cloud = pv.PolyData(point_data)
    point_cloud.plot(eye_dome_lighting=True)

    #create structured grid
    x_space=1.5
    y_space=1.5
    extra=0
    box = point_cloud.outline()
    bounds = box.bounds
    print("bounds: ",bounds)
    dimensions=(int(bounds[1]-bounds[0])+1,int(bounds[3]-bounds[2])+1,int(bounds[5]-bounds[4])+2)
    origin=(int(bounds[0]),int(bounds[2]),int(bounds[4])-1)
    print("dimensions: ",dimensions)
    spacing = (dimensions[0]/15,dimensions[1]/15,dimensions[2])
    print("spacing: ",spacing)
    grid = pv.UniformGrid(dims = (int((dimensions[0]+1)/x_space),int((dimensions[1]+1)/y_space),2),origin=origin, spacing = (x_space,y_space,dimensions[2]))
    #grid.plot(show_edges=True)


    
    points_to_search=point_data
    for i in range(0,grid.number_of_cells-1):
        #points_in_cell=[]
        cell_bounds = grid.cell_bounds(i)
        xmin=cell_bounds[0]
        xmax=cell_bounds[1]
        ymin=cell_bounds[2]
        ymax=cell_bounds[3]
        zmin=cell_bounds[4]
        zmax=cell_bounds[5]
        xy_center = np.array([(xmax+xmin)/2,(ymax+ymin)/2])
        points_found=False
        print("cells to search: ",grid.number_of_cells-i)
        #print("points to search: ",points_to_search.shape[0])
        first_point=True
        for point in points_to_search:
            point=np.array(point,ndmin=2)
            if point[0][0]>=xmin and point[0][0]<=xmax and point[0][1]>=ymin and point[0][1]<=ymax and point[0][2]>=zmin and point[0][2]<=zmax:
                #print("point:", point)
                if points_found==False:
                    points_in_cell=point
                else:
                    #print("point added on: ",point)
                    #print("points inarray: ", points_in_cell)
                    points_in_cell=np.append(points_in_cell, point,axis=0)
                points_found=True
            else:
                if first_point==False:
                    points_to_search=np.append(points_to_search,point,axis=0)
                else:
                    points_to_search=np.array(point,ndmin=2)
                first_point=False
            
        if points_found==True:
            if points_in_cell.size<2:
                mean=points_in_cell
            else:
                 mean=np.array(np.mean(points_in_cell,axis=0),ndmin=2)
            mean=np.array(np.append(xy_center,mean[0][2]),ndmin=2)
            #print("points in cell: ", points_in_cell)
            #print("mean: ", mean)
            try:
                simplified_data = np.append(simplified_data, mean,axis=0)
                #print("simplified data in try: ",simplified_data)
            except:
                simplified_data=np.array(mean,ndmin=2)
        else:
            xy_center=np.array(xy_center,ndmin=2)
            try:
                empty_cells = np.append(empty_cells, xy_center,axis=0)
                #print("simplified data in try: ",simplified_data)
            except:
                empty_cells=np.array(xy_center,ndmin=2)
                
        #print("empty cells: ",empty_cells)

    xmin=min(empty_cells[:,0])
    ymin=min(empty_cells[:,1])
    xmax=max(empty_cells[:,0])
    ymax=max(empty_cells[:,1])
    x_space=x_space
    y_space=y_space
    extra=10
    extra_x = np.append(np.arange(xmin-extra,xmin,x_space),np.arange(xmax,xmax+extra,x_space))
    print("extra x: ",extra_x)
    extra_y = np.append(np.arange(ymin-extra,ymin,y_space),np.arange(ymax,ymax+extra,y_space))
    all_x = np.arange(xmin,xmax,x_space)
    all_y = np.arange(ymin-extra,ymax+extra,y_space)
    extra_xy=np.array([xmin-extra,ymin-extra],ndmin=2)
    for x in extra_x:
        for y in all_y:
            extra_xy=np.vstack([extra_xy,np.array([x,y],ndmin=2)])
    for y in extra_y:
        for x in all_x:
            extra_xy=np.vstack([extra_xy,np.array([x,y],ndmin=2)])
    #empty_cells = np.append(empty_cells,extra_xy,axis=0)

    simplified_data_flat = np.append(point_data[:,0:2],np.array(np.zeros((point_data.shape[0],1)),ndmin=2),axis=1)
    print("simp flat: ",simplified_data_flat)
    simplified_cloud=pv.PolyData(simplified_data_flat)

    for cell in extra_xy:
        cell=np.append(cell,0)
        i = simplified_cloud.find_closest_point(cell)
        closest_point=simplified_cloud.points[i]
        distance = np.linalg.norm(cell-closest_point)
        if distance<3:
            try:
                extra_cells=np.vstack([extra_cells,cell]) 
            except:
                extra_cells=np.array(cell,ndmin=2)

    for cell in empty_cells:
        cell=np.append(cell,0)
        i = simplified_cloud.find_closest_point(cell)
        closest_point=simplified_cloud.points[i]
        distance = np.linalg.norm(cell-closest_point)
        if distance<2.5:
            try:
                fill_cells=np.vstack([fill_cells,cell]) 
            except:
                fill_cells=np.array(cell,ndmin=2)

    print("point: ",closest_point)
    print("cell: ",cell)
    print("extra_cell:", empty_cells.shape)

    surface_fit_points = fit_surface(point_data, fill_cells,fill_cells)
    all_points=np.vstack([surface_fit_points,simplified_data])
    point_data=remove_outliers(point_data,point_data)
    point_data = np.vstack([point_data,surface_fit_points])


    
    point_cloud=pv.PolyData(point_data)
    pl= pv.Plotter()
    pl.add_mesh(point_cloud,color='b')
    pl.show()

    point_cloud.plot()
    

    return point_cloud

def construct_surface_2D(point_cloud): 
    surf = point_cloud.delaunay_2d()
    #surf.plot(show_edges=True)

    #surf.plot(show_edges=True)
    surf.smooth(n_iter=150,inplace=True)
    #surf.plot(show_edges=True)
    
    surf_boundary = surf.extract_feature_edges(feature_angle=150)
    print("surf bound: ",surf_boundary)
    #surf_boundary.plot(show_edges=True)


    

    return surf, surf_boundary

def combine_surfaces(surf0, surf1, surf2):
    #ex_length = np.max(surf0[:,2])-np.min(surf1[:,2])
    surf0.compute_normals(inplace=True)
    surf1.compute_normals(inplace=True)
    surf2.compute_normals(inplace=True)

    surf0 = surf0.subdivide(nsub=2)
    surf1 = surf1.subdivide(nsub=2)

    bounds=pv.PolyDataFilters.merge(surf0,surf1).bounds
    bounds1=surf1.bounds
    bounds0=surf0.bounds

    #Make rectilinear grid
    cell_size=0.5
    x = np.arange(bounds[0],bounds[1])
    y = np.arange(bounds[2],bounds[3])
    z = np.arange(bounds1[4],bounds0[5])
    dataset = pv.RectilinearGrid(x,y,z)

    p = pv.Plotter()
    p.add_mesh(surf0, color='r', style='wireframe')
    p.add_mesh(surf1, color='b', style='wireframe')
    #p.add_mesh(dataset,color='gold',show_edges=True,opacity=0.75)
    #p.show()
    
    if surf1.active_normals[5,2]>0:
        surf1.flip_normals()
    dorsal_ex=surf1.extrude([0,0,8],capping=True)
    #dorsal_ex.plot_normals(mag=5)
    dorsal_ex.compute_normals(inplace=True)


    if surf0.active_normals[5,2]<0:
        surf0.flip_normals()
    volar_ex=surf0.extrude([0,0,-8],capping=True)
    volar_ex.compute_normals(inplace=True)
    #volar_ex.plot_normals(mag=5)

    if surf2.active_normals[5,2]<0:
        surf2.flip_normals()
    volar_ex2=surf2.extrude([0,0,10],capping=True)
    volar_ex2.compute_normals(inplace=True)
    #volar_ex.plot_normals(mag=5)

    #surf0.plot_normals(show_edges=True,mag=5)
    #surf1.plot_normals(show_edges=True,mag=5)
 
    clipped_datasetv = dataset.clip_surface(volar_ex, invert=True)
    clipped_datasetd = clipped_datasetv.clip_surface(dorsal_ex, invert=True)
    clipped_dataset = clipped_datasetd.clip_surface(volar_ex2, invert=True)
    #clipped_datasetde = clipped_datasetd.clip_surface(dorsal_ex,invert=True)
    #clipped_dataset = clipped_datasetde.clip_surface(volar_ex,invert=True)

    p = pv.Plotter()
    p.add_mesh(clipped_dataset, color='gold',show_edges=True,opacity=0.75)
    p.add_mesh(volar_ex,style='wireframe',color ='purple')
    p.add_mesh(dorsal_ex,style='wireframe',color = 'green')
    p.show()

    muscle_mesh=clipped_dataset.extract_surface()
    muscle_mesh.plot(show_edges=True)
    muscle_mesh.smooth(n_iter=100,inplace=True)
    muscle_mesh.plot(show_edges=True)
 
    print("manifold? ",muscle_mesh.is_manifold)
    print("volume: ", muscle_mesh.volume)
    print("n points: ",muscle_mesh.number_of_points)

    pv.save_meshio(subject_path + '\\APB.ply',muscle_mesh)
    #pv.save_meshio(subject_path + '\\FPB.stl',muscle_mesh)

    return muscle_mesh

def interp_points(mesh):
    bounds = mesh.bounds
    x = np.arange(bounds[0],bounds[1],0.5)
    y = np.arange(bounds[2],bounds[3],0.5)
    z = np.arange(bounds[4],bounds[5],0.5)
    points = pv.RectilinearGrid(x,y,z)
    sample = mesh.interpolate(points)
    mesh=mesh.triangulate()
    new_mesh=mesh.subdivide(nsub=1)
    print("new mesh: ",new_mesh)
    cloud = pv.PolyData(sample.points)
    pl = pv.Plotter()
    pl.add_mesh(mesh, style='wireframe',color='r')
    pl.add_mesh(new_mesh,show_edges=True,color='b')
    pl.show()


    return new_mesh.points  



subject_path = 'C:\\Users\\jocel\\OneDrive\\Desktop\\20220607 Formal Data Collection\\Sub00000009\\Trial 1\\'
volar_filename = subject_path +'APBvolarpoints.txt'
volar2_filename = subject_path + 'APBvolarPoints.txt'
dorsal_filename = subject_path + 'APBDorsalPoints.txt'
side = 0
volar_pts = simplify_points(volar_filename)
volar2_pts = simplify_points(volar2_filename)
side = 1
dorsal_pts = simplify_points(dorsal_filename)
p = pv.Plotter()
p.add_mesh(volar_pts, color='red')
p.add_mesh(volar2_pts, color='purple')
p.add_mesh(dorsal_pts, color='blue')
p.show()
volar_surf, volar_bound = construct_surface_2D(volar_pts)
volar2_surf, volar_bound = construct_surface_2D(volar2_pts)
dorsal_surf, dorsal_bound = construct_surface_2D(dorsal_pts)
mesh = combine_surfaces(volar_surf, dorsal_surf, volar2_surf) 
dense_points = interp_points(mesh) 
np.savetxt(subject_path + 'APB Dense Point Cloud.txt',dense_points) 










