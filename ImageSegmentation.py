import matplotlib
import scipy.interpolate 
from matplotlib import image
from matplotlib.widgets import Button
from skimage import io, feature, filters, morphology
from skimage.color import rgb2gray
from skimage.morphology import flood_fill
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from random import randint
from scipy.optimize import curve_fit
import pandas as pd
import cv2


#PARAMETERS:
CANNY_sigma = 3
CANNY_lowt = 0.04
CANNY_hight = 0.12   
HYST_lowt_vals = [0.35,0.35,0.3,0.35]
HYST_hight_vals = [0.65,0.68,0.6,0.63]
cutoff_radius = 10
side = 'dorsal'
muscle = 'FPB'

subject_path = 'Trial 1/'
TEI_path = 'TEI.txt'
APB_path = subject_path + 'APB Dense Point Cloud.txt'
FPB_path = subject_path + 'FPB Dense Point Cloud.txt'
OPP_path = subject_path + 'OPP Dense Point Cloud.txt'




class Scan:
    def __init__(self,folder,start,stop,seg_range,current_muscle,other_muscles,TRI_all):
        self.current_muscle = current_muscle
        self.seg_range=[x-start for x in seg_range]
        self.muscles = other_muscles
        self.start = start
        self.stop = stop
        self.folder =  folder
        self.scan_folder = subject_path + folder
        self.files = os.listdir(self.scan_folder)
        self.images=[]
        self.skin_points = pd.DataFrame(columns=['X','Y','Z'])
        self.volar_points = pd.DataFrame(columns=['X','Y','Z'])
        self.dorsal_points = pd.DataFrame(columns=['X','Y','Z'])
        for i in range(self.start,self.stop):
            filename = self.scan_folder + '//' + self.files[i]
            img = rgb2gray(io.imread(filename))[60:550,528:1250]
            TRI = TRI_all[int(self.folder),i]
            current_img = self.usImage(img,TRI,self.images,int(self.folder))
            self.images.append(current_img)

    #Get volar skin points 
    def skin_reconstruct(self):
        for img in self.images:
            img.get_skin_surf()
            if img.skin_surf[0,0]==0 and img.skin_pts_RCS[0,0] == 0:
                continue
            self.skin_points._append(img.skin_pts_RCS)
        self.get_max_skin_point()
        

    def segment_images(self):
        for i,img in enumerate(self.images[self.seg_range[0]:self.seg_range[1]]):
            print("scan: ", self.folder, "image: ",self.start+i," out of ",self.stop)
            if muscle == 'APB':
                img.get_muscle_surf()
            img.project_muscle(self.current_muscle,self.other_muscles)
            img.crop()
            if i == 0 or img.redo == True:
                img.get_border_by_click()
            else:
                prev_img = self.images[self.seg_range[0]+i-1]
                img.get_border_by_tracking(prev_img,prev_img.dorsal_pts)

    
            current_img=img
            self.muscles = current_img.segment(self.shift,self.muscles,self.seg_range)
            if muscle != 'APB':
                current_img.skip_v = True

            #create array of all RCS points in this scan
            if current_img.skip_v == False:
                self.volar_points = self.volar_points.append(current_img.volar_pts_RCS)


            if current_img.skip_d == False:
                self.dorsal_points = self.dorsal_points.append(current_img.dorsal_pts_RCS) 


    #Fits skin points to 3D polynomial and gets the max point (highest z-value)
    def get_max_skin_point(self):
        point_data = self.skin_points
        def function(data, a0, a1,a2, a3, a4, a5):
            x = data[0]
            y = data[1]
        
            return a0+a1*x+a2*y+a3*x*y+a4*x**2+a5*y**2

        x_data = []
        y_data = []
        z_data = []
        for item in point_data:
            x_data.append(item[0])
            y_data.append(item[1])
            z_data.append(item[2])
        # get fit parameters from scipy curve fit
        parameters, covariance = curve_fit(function, [x_data, y_data], z_data)

        # create surface function model
        # setup data points for calculating surface model
        model_x_data = np.linspace(min(x_data)-10, max(x_data)+10, 30)
        model_y_data = np.linspace(min(y_data)-10, max(y_data)+10, 30)
        # create coordinate arrays for vectorized evaluations
        X, Y = np.meshgrid(model_x_data, model_y_data)

        # calculate Z coordinate array
        Z = function(np.array([X, Y]), *parameters)
        x,y,z =X.flatten(), Y.flatten(), Z.flatten()
        points = np.stack([x,y,z],axis=1)
        self.max_skin_point = points[np.where(points[:,2] == np.max(points[:,2]))]

    #Shift points according to skin
    #This is to help account for slight movement of the hand between scans 
    #(doesn't account for rotation, but works bc there is very little difference between scans)
    def shift_points(self,other_msp):
        shift = other_msp-self.max_skin_point
        if muscle == 'APB':
            self.volar_points += shift 
        self.dorsal_points += shift
        self.skin_points += shift 

    class usImage:
        def __init__(self,img,TRI,images,scan_n):
            self.img = img
            self.TRI = TRI
            self.images = images
            self.n=scan_n
            self.hyst_lowt = HYST_lowt_vals[scan_n]
            self.hyst_hight = HYST_hight_vals[scan_n]
            self.skin_surf = []
            self.skin_pts_RCS = []
            self.redo,self.skip_d,self.skip_v = False,False,False
            self.x_adj, self.y_adj = 0,0
            self.dorsal_pts = pd.DataFrame(columns=['X','Y'])

        #Show previous muscles on ultrasound image
        def project_muscle(self,current_muscle,muscles):

            def RCS_to_ICS(point,TIR):
                point = np.append(point,[1])
                point_ICS = np.delete(np.matmul(TIR,point),3)
                point_ICS = np.delete(point_ICS,2)/0.0666666667
                
                return point_ICS

            TRI = self.TRI
            TIR = np.linalg.inv(TRI)
            shifted_current = current_muscle + self.shift
            for muscle in muscles:
                ind_del = []
                i=0
                shifted_muscle = muscle + self.shift

                #Convert muscle points from 3D CS to 2D Image CS
                other_musc = []
                for point in shifted_muscle:
                    point_ICS = RCS_to_ICS(point,TIR)
                    if abs(point_ICS[2])<5: 
                        point_ICS = np.delete(point_ICS,2)/0.066666666666666667                    
                        other_musc.append(point_ICS)
                        ind_del.append(i)
                    i += 1
                if len(ind_del)>0:
                    muscle = np.delete(muscle,ind_del,axis=0)
                self.pts_ICS = np.array(other_musc,ndmin=2)

            curr_musc=[]
            #Get muscle points present in the current image (where Z~=0)                        
            for point in shifted_current:
                point_ICS = RCS_to_ICS(point)
                if abs(point_ICS[2])<5: 
                    curr_musc.append(point_ICS)
            self.current_pts_ICS=np.array(curr_musc,ndmin=2) 

        def get_skin_surf(self):
            hyst = self.apply_hyst_thresh()
            footprint = morphology.disk(2)
            res_bright = morphology.white_tophat(hyst, footprint)
            indices_bright = np.where(res_bright == True)
            indices_bright = np.stack([indices_bright[0],indices_bright[1]],axis=1)
            for i in indices_bright:
                hyst[i[0],i[1]] = False

            hyst_T = np.transpose(hyst)
            for i,col in enumerate(hyst_T):
                bright_found=False
                for j,pixel in enumerate(col):
                    pixel_coord = [i,j]
                    if j<80:
                        j+=1
                        continue
                    if pixel == True and bright_found == False:
                        #only add to skin_surf if j is within 10 pixels of last j (skips bright particles/noise above skin surface)
                        if len(self.skin_surf)>0:
                            if abs(self.skin_surf[-1][1]-j)<20:
                                self.skin_surf.append(pixel_coord)
                                self.skin_pts_RCS.append(ICS_to_RCS([i,j],self.TRI))
                                bright_found=True
                        else:
                            self.skin_surf.append(pixel_coord)
                            self.skin_pts_RCS.append(ICS_to_RCS([i,j],self.TRI))
                            bright_found=True
            self.skin_surf = pd.DataFrame(self.skin_surf,columns = ['X','Y'])
            self.skin_pts_RCS = pd.DataFrame(self.skin_pts_RCS,columns = ['X','Y','Z'])

            #get max skin surface point
            skin_peakidx = self.skin_surf['Y'].idxmin()
            self.skin_peak = self.skin_surf.loc[skin_peakidx]

        
        #Get the volar muscle border points
        def get_muscle_surf(self):
            #apply hysteresis threshold
            hyst = self.apply_hyst_thresh()
            hyst_T = np.transpose(hyst)

            self.skin_surf = np.array(self.skin_surf, ndmin=2)
            #Go along the skin surface and search for the muscle surface below it
            l2d_edge=[]
            for point in self.skin_surf:
                j=point[0]
                i=point[1]
                for pixel in hyst_T[j,point[1]:]:
                    if pixel == False:
                        pixel_coord = [j,i]
                        l2d_edge.append(pixel_coord)
                        break
                    i+=1
            l2d_edge = np.array(l2d_edge,ndmin=2)

            #Apply canny edge detection
            canny = feature.canny(self.img,5.5,low_threshold = 0.05, high_threshold = 0.1)

            volar_points = []
            #Remove outliers in volar muscle surface
            for point in l2d_edge:
                skin_point = self.skin_surf[np.where(self.skin_surf[:,0] == point[0])][0]
                upper_bound = ((skin_point[1]+point[1])/2+point[1])/2
                lower_bound = point[1]+50
                i,j=point[1],point[1]
                while i>upper_bound or j<lower_bound:
                    if j>canny.shape[0]-1 or i<0:
                        break
                    if i > upper_bound:
                        pixel = canny[i,point[0]]
                        if pixel == True:
                            volar_point = [point[0],i]
                            break
                    i+=-1
                    if j < lower_bound:
                        pixel = canny[j,point[0]]
                        if pixel == True:
                            volar_point = [point[0],j]
                            break
                    j+=1
                
                #If border point found, add it to volar_pts
                if pixel == True:
                    volar_points.append(volar_point)    
            self.volar_pts = pd.DataFrame(volar_points,columns = ['X','Y'])
            volar_np = np.array(volar_points,ndmin=2)

            self.muscle_f = np.polyfit(volar_np[:,0],volar_np[:,1],deg = 2)



            #take curve fit to volar points of previous image. If volar points of current image aren't within 10 pixels of fitting curve, remove.
            """if self.images.index(self)>self.seg_range[0]:
                self.current_img_index = self.images.index(self)
                f = self.images[self.current_img_index-1].muscle_f
            else: 
                f=self.muscle_f
            yi = np.polyval(f,self.volar_pts[:,0])
 
            inidices_to_del = []
            for i,y in enumerate(self.volar_pts['Y']):
                if abs(y-yi[i])>45:
                    inidices_to_del.append(i)
            self.volar_pts = self.volar_pts.drop(inidices_to_del,axis=0)"""

        def get_border_by_click(self):
            #set img to cropped img and adjust volar points to fit cropped image
            adjust=[self.x_adj,self.y_adj]
            
            if muscle == 'APB':
                volar_pts = self.volar_pts - [self.x_adj,self.y_adj]

                #Interpolate cutoff points 
                f = scipy.interpolate.interp1d(volar_pts[:,0],volar_pts[:,1])
                xi = np.arange(np.min(volar_pts[:,0]),np.max(volar_pts[:,0]),1)
                yi = f(xi)
                interp_vp = np.stack([xi,yi],axis=1)

            img = self.img

            #Apply Canny Edge filter
            edges = feature.canny(img,CANNY_sigma,low_threshold = CANNY_lowt, high_threshold = CANNY_hight)
            self.edges = edges  
            #get points near muscle boundary, if first image: manual selection; otherwise, use previous b2mpeak
            #if manual selection, cutoff is set to volar points. If using previous b2mpeak, cutoff is set to previously selected points
            clicked_points = click_point_coordinate(img*255)
            ind = np.lexsort((clicked_points[:,1],clicked_points[:,0]))
            clicked_points = clicked_points[ind]

            if muscle == 'APB':
                #Remove points outside of cutoff + cutoff radius
                for point in interp_vp:
                    #skips points if they are outside image bounds. idk why some points are out of bounds. I should fix this
                    if point[0]>edges.shape[1]-1:
                        continue
                    for i in range(0,edges.shape[0]-1):
                        if i<point[1] - cutoff_radius:
                            edges[i,int(point[0])]=False 

            #edge_coords = array_to_xy(edges,True)

            #Find closest edge points to each clicked point
            #calculate distance between every clicked point and every edge coord, and find edge closest to clicked point
            closest_edges = []
            img_borders=[]
            #Convert binary Canny Edge image to grayscale image
            edges_remember = binary_to_gs(edges)
            i=0
            for point in clicked_points:
                edges_gs = binary_to_gs(edges)
                closest_edges.append(find_closest_edge(point,edges))
                edge = find_closest_edge(point,edges)
                if len(edge)<2:
                    continue
                if edges_remember[edge[0],edge[1]] == 75:
                    i-=1
                    continue
                edges_remember = flood_fill(edges_remember,(edge[0],edge[1]),75)
                edges_gs = flood_fill(edges_gs,(edge[0],edge[1]),75)
                img_borders.append(get_border(edges_gs,side))
                edge_pts = array_to_xy(edges_gs,75)
                edge_pts['edge_n'] = i
                self.dorsal_pts = self.dorsal_pts.append(edge_pts)
                i+=1
            self.dorsal_pts = self.dorsal_pts.set_index('edge_n')

            if len(closest_edges)<1:
                print("redo: no close edges found.")
                self.redo = True
                return

        def get_border_by_tracking(self,prev_img,prev_points):
            edge0 = feature.canny(prev_img,CANNY_sigma,low_threshold = CANNY_lowt, high_threshold = CANNY_hight)     
            edge1 = feature.canny(self.img,CANNY_sigma,low_threshold = CANNY_lowt, high_threshold = CANNY_hight) 
            edge0_gs, edge1_gs = binary_to_gs(edge0), binary_to_gs(edge1)
            edge0_u8, edge1_u8 = (edge0_gs/255).astype('uint8'), (edge1_gs/255).astype('uint8')

            p0 = prev_points.to_numpy(dtype=np.float32)

            p1 = cv2.calcOpticalFlowPyrLK(edge0_u8,edge1_u8,p0,None)[0]
            print('p1: ',p1)

            points = []
            dorsal = []
            for p in p1:
                p=[int(p[0]),int(p[1])]
                point = find_closest_edge(p,edge1)
                if len(points) < 0:
                    for pt in points:
                        if pt == point:
                            break
                    if pt == point:
                        continue
                points.append(point)

            img_borders=[]
            #Convert binary Canny Edge image to grayscale image
            edges_remember = binary_to_gs(edge0)
            i=0
            for edge in points:
                if edges_remember[edge[0],edge[1]] == 75:
                    i-=1
                    continue
                edges_remember = flood_fill(edges_remember,(edge[0],edge[1]),75)
                edges_gs = flood_fill(edge1_gs,(edge[0],edge[1]),75)
                img_borders.append(get_border(edges_gs,side))
                edge_pts = array_to_xy(get_border(edges_gs,side),75)
                self.dorsal_pts = self.dorsal_pts.append(pd.DataFrame(edge_pts,columns=['X','Y']))
                #self.dorsal_pts['edge_n'] = int(i)
                i+=1
            #self.dorsal_pts = self.dorsal_pts.set_index('edge_n')


            #Adjust extracted dorsal points to Image Coordinate System
            self.dorsal_pts['X'] += self.x_adj
            self.dorsal_pts['Y'] += self.y_adj

            self.b2mpeak = self.dorsal_pts
            self.redo = False

        def crop_image(self):
            print(self.skin_peak)
            ymin, ymax = self.skin_peak['Y'], self.skin_peak['Y']+350 
            self.cropped_img = self.img[ymin:ymax,:]
            
            sigma=5
            #blurred = skimage.filters.gaussian(self.cropped_img, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
            blurred = filters.apply_hysteresis_threshold(self.cropped_img,0.4,0.75)
            blurred = binary_to_gs(blurred)
            min_point = np.where(blurred[:,self.skin_peak['X']]==np.min(blurred[5:,self.skin_peak['X']]))[0][0]
            blurred=blurred*255
            flood = flood_fill(blurred,(min_point,self.skin_peak['X']),255,tolerance=100-blurred[min_point,self.skin_peak['X']])
            skin_flood = flood_fill(blurred,(self.skin_peak['Y']-ymin,self.skin_peak['X']+2),255,tolerance=35)

            muscle_coords = np.where(flood == 255)
            skin_coords = np.where(skin_flood == 255)
            muscle_tip = np.min(muscle_coords[1])

            xmin, xmax = muscle_tip-10, max(self.skin_surf['X'])-30
            if xmin<0:
                xmin=0
            print(xmin,xmax,ymin,ymax)

            self.cropped_img = self.img[ymin:ymax,xmin:xmax]
            self.x_adj = xmin
            self.y_adj = ymin

            redo = False
            return redo

        #display plot with extracted points
        def show_plot(self):
            plt.imshow(self.img, cmap='gray')
            if muscle == 'APB':
                plt.scatter(self.volar_pts[:,0],self.volar_pts[:,1],color='red',s=0.5)
            plt.scatter(self.dorsal_pts['X'],self.dorsal_pts['Y'],color='blue',s=0.5)
            #plt.scatter(self.pts_ICS[:,0],self.pts_ICS[:,1], color='green', s=0.3 )
            #plt.scatter(self.current_pts_ICS[:,0],self.current_pts_ICS[:,1], color='purple', s=0.3 )
            def press_redo(event):
                self.redo = True 
                print("redo: redo button pressed")
            def press_skip_d(event):
                self.skip_d = True 
                print("skipping. no dorsal points extracted.")
            def press_skip_v(event):
                self.skip_v = True 
                print("skipping. no dorsal points extracted.")
            
            #Redo button
            axes = plt.axes([0.7, 0.05, 0.1, 0.075])
            bredo = matplotlib.widgets.Button(axes,'Redo')
            bredo.on_clicked(press_redo)

            #Do not use dorsal points button
            axes = plt.axes([0.4, 0.05, 0.2, 0.075])
            bskip_d = matplotlib.widgets.Button(axes,'Skip dorsal points')
            bskip_d.on_clicked(press_skip_d)

            #Do not use volar points button
            axes = plt.axes([0.1, 0.05, 0.2, 0.075])
            bskip_v = matplotlib.widgets.Button(axes,'Skip volar points')
            bskip_v.on_clicked(press_skip_v)
            plt.show()

        def apply_hyst_thresh(self):
            img = self.img
            #apply hysteresis threshold filter
            hyst = filters.apply_hysteresis_threshold(img, self.hyst_lowt, self.hyst_hight)

            #remove bright noise
            footprint = morphology.disk(1)
            res_bright = morphology.white_tophat(hyst, footprint)
            indices_bright = np.where(res_bright == True)
            indices_bright = np.stack([indices_bright[0],indices_bright[1]],axis=1)
            for i in indices_bright:
                hyst[i[0],i[1]] = False

            #remove dark noise
            hyst_invert = ~hyst
            footprint = morphology.disk(4)
            res_dark = morphology.white_tophat(hyst_invert, footprint)
            indices_dark = np.where(res_dark == True)
            indices_dark = np.stack([indices_dark[0],indices_dark[1]],axis=1)
            for i in indices_dark:
                hyst[i[0],i[1]] = True

            return hyst


#Convert points from 2D image coordinates to 3D robot coordinates
def ICS_to_RCS(pt,TRI):
    point = [pt[0]*0.066666666667,pt[1]*0.066666666667]
    point = np.array(point,ndmin=2)
    if point.shape[0]>2:
        point=np.append(point,[1])
    else:
        point = np.append(point,[0,1])
    point_RCS = np.delete(np.matmul(TRI,point),3)

    return point_RCS

def get_border(img,side):
    #Get top points only
    flood_T = np.transpose(img)
    if side == 'dorsal':
        for i,col in enumerate(flood_T):
            top_found = False
            lower_pixel_searched = False
            for j,pixel in enumerate(col):
                if top_found == True and lower_pixel_searched == True:
                    img[j,i] = 0
                if top_found == True:
                    lower_pixel_searched = True
                if pixel == 75:
                    top_found = True

    #gets bottom points only
    else:
        for col in flood_T:
            col = np.flip(col)
            top_found = False
            lower_pixel_searched = False
            j=len(col)-1
            for pixel in col:
                if top_found == True and lower_pixel_searched == True:
                    img[j,i] = 0
                if top_found == True:
                    lower_pixel_searched = True
                if pixel == 75:
                    top_found = True
                j-=1
            i+=1
    
    return img

def find_closest_edge(point,edges):
    for i in range(0,15):
        for j in range(-i-1,i+1):
            for k in [-i-1,i+1]:
                npoint = [point[0]+i,point[1]+j]
                if edges[npoint[1],npoint[0]] == True:
                    return [npoint[1],npoint[0]]
    return []



def click_point_coordinate(img):
    #setting up a tkinter canvas
    root = Tk()
    canvas = Canvas(root,width = img.shape[1], height = img.shape[0])

    #adding the image
    clicky_img = ImageTk.PhotoImage(image=Image.fromarray(img))
    label = Label(image=clicky_img)
    label.image = clicky_img # keep a reference!
    label.pack()

    canvas.create_image(0,0,image=clicky_img,anchor="nw")

  
    #function to be called when mouse is clicked
    def printcoords(event):
        global points
        #outputting x and y coords to console
        point = np.array([event.x,event.y],ndmin=2)
        try:
            points.append(point)
        except: 
            points=[point]

    #mouseclick event
    label.bind("<Button 1>",printcoords)
    label.pack()

    root.mainloop()


    points_clicked = np.array(points[0],ndmin=2)
    points.pop(0)
    for i in range(0,len(points)):
        points_clicked = np.vstack([points_clicked,points[0]])
        points.pop(0)

    return points_clicked

def binary_to_gs(img):
    #convert edges to GrayScale
    img_gs = np.zeros([img.shape[0],img.shape[1]])
    i=0
    for row in img:
        j=0
        for pixel in row:
            if pixel == True:
                img_gs[i,j] = 255
            else:
                img_gs[i,j] = 0
            j+=1
        i+=1

    return img_gs

def array_to_xy(img,val):
    #convert table data to a two column array of xy values
    xy_coords=[]
    for i,row in enumerate(img):
        for j,pixel in enumerate(row):
            if pixel == val:
                xy = [j,i]
                xy_coords.append(xy)
    xy_coords = pd.DataFrame(xy_coords, columns = ['X','Y'])

    return xy_coords

def get_TRI_array(TRE_folder,TEI_filename):
    TEI = np.loadtxt(TEI_filename, dtype=float)
    TRI_list=[]
    for i in range(0,4):
        TRE_filename = TRE_folder + 'TRE_' + str(i) +'.txt'
        TRE_data = np.loadtxt(TRE_filename, dtype=float)
        n_frames = int(TRE_data.shape[0]/4 )
        TRI_scan = np.array(np.matmul(TRE_data[0:4],TEI),ndmin=3)
        
        for i in range(1,n_frames):
            TRE = TRE_data[i*4:i*4+4]
            #TRE = np.matmul(T0n,TRE)
            TRI = np.array(np.matmul(TRE,TEI),ndmin=3)
            TRI_scan = np.concatenate((TRI_scan,TRI),axis=0)
        TRI_list.append(TRI_scan) 
  
    TRI_all = np.stack((TRI_list[0],TRI_list[1],TRI_list[2],TRI_list[3]),axis=0)
    
    return TRI_all




def main():
    volar_pts, dorsal_pts = pd.DataFrame(columns = ['X','Y','Z']), pd.DataFrame(columns = ['X','Y','Z'])
    #4d array of all TRIs
    TRI_all = get_TRI_array(subject_path,TEI_path)
    APB,FPB,OPP = pd.DataFrame(columns=['X','Y','Z']), pd.DataFrame(columns=['X','Y','Z']), pd.DataFrame(columns=['X','Y','Z'])
    muscles = [APB,FPB,OPP]
    muscle_paths = [APB_path,FPB_path,OPP_path]
    for i,muscle in enumerate(muscles):
        try:
            muscle = pd.read_excel(muscle_paths[i])
        except:
            continue
    
    current_muscle = dorsal_pts
    
    s0=Scan('0',6,42,[20,35],current_muscle,muscles,TRI_all) 
    s1=Scan('1',5,57,[10,55],current_muscle,muscles,TRI_all) 
    s2=Scan('2',0,62,[15,45],current_muscle,muscles,TRI_all)
    s3=Scan('3',19,62,[20,45],current_muscle,muscles,TRI_all) 
    scans = [s2,s0]
    for scan in [s0,s2]:
        print(scan.scan_folder)
        scan.skin_reconstruct()
        print(scan.max_skin_point)


    for scan in scans:
        scan.shift = (scan.max_skin_point-s0.max_skin_point)
        scan.segment_images()
        scan.shift_points(s0.max_skin_point)
        volar_pts = volar_pts.append(scan.volar_points)
        dorsal_pts = dorsal_pts.append(scan.dorsal_points)
        current_muscle = dorsal_pts

    with pd.ExcelWriter("pointcloud.xlsx") as writer:
        dorsal_pts.to_excel(writer, sheet_name=muscle + '_' + side)
        if muscle == 'APB':
            volar_pts.to_excel(writer, sheet_name='APB_volar')

main()
