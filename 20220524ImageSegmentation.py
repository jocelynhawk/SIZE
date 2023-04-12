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


#PARAMETERS:
CANNY_sigma = 3
CANNY_lowt = 0.04
CANNY_hight = 0.12   
HYST_lowt_vals = [0.35,0.35,0.3,0.35]
HYST_hight_vals = [0.65,0.68,0.6,0.63]
cutoff_radius = 10
side = 'dorsal'
muscle = 'APB'


#Convert points from 2D image coordinates to 3D robot coordinates
def ICS_to_RCS(pt,TRI):
    point = pt * 0.066666666667
    point = np.array(point)
    if point.shape[0]>2:
        point=np.append(point,[1])
    else:
        point = np.append(point,[0,1])
    point_RCS = np.delete(np.matmul(TRI,point),3)
    point_RCS = pd.DataFrame([point_RCS[0],point_RCS[1],point_RCS[2]],columns=['X','Y','Z'])


    return point_RCS

class Scan:
    def __init__(self,folder,start,stop,seg_range):
        self.seg_range=[x-start for x in seg_range]
        self.muscles = muscles
        self.start = start
        self.stop = stop
        self.folder = folder
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
        for img in self.images[self.seg_range[0]:self.seg_range[1]]:
            current_img=img
            self.muscles = current_img.segment(self.shift,self.muscles,self.seg_range)
            if muscle != 'APB':
                current_img.skip_v = True

            #create array of all RCS points in this scan
            if current_img.skip_v == False:
                self.volar_points._append(current_img.volar_pts_RCS)


            if current_img.skip_d == False:
                self.dorsal_points._append(current_img.dorsal_pts_RCS) 


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
            self.current_img_index=len(images)
            self.skin_surf = pd.DataFrame(columns=['X','Y'])
            self.skin_pts_RCS = pd.DataFrame(columns=['X','Y','Z'])

        #segment image
        def segment(self,shift,muscles,seg_range):
            print("scan: ", self.n, "image: ",self.current_img_index," out of ",seg_range[1])

            self.seg_range=seg_range
            self.muscles = muscles
            self.shift=shift

            self.project_muscle()
            self.get_volar_pts()

            self.redo,first,self.skip_d,self.skip_v = False,True,False,False
            while self.redo == True or first == True:
                self.get_dorsal_pts()
                first = False
                if self.redo == False:
                    self.show_plot()
                if self.skip_d == True or self.skip_v == True:
                    break
            
            #APB uses volar muscle points
            if muscle =='APB':
                if self.volar_pts.shape[0]>1:
                    self.volar_pts_RCS = self.ICS_to_RCS(self.volar_pts)
            self.dorsal_pts_RCS = self.ICS_to_RCS(self.dorsal_pts)
            return self.muscles

        #Show previous muscles on ultrasound image
        def project_muscle(self):
            TRI = self.TRI
            TIR = np.linalg.inv(TRI)
            shifted_current = current_muscle + self.shift
            for muscle in self.muscles:
                ind_del = []
                i=0
                shifted_muscle = muscle + self.shift

                #Convert muscle points from 3D CS to 2D Image CS
                for point in shifted_muscle:
                    point = np.append(point,[1])
                    point_ICS = np.delete(np.matmul(TIR,point),3)
                    if abs(point_ICS[2])<0.1: 
                        point_ICS = np.delete(point_ICS,2)/0.066666666666666667                    
                        try:
                            self.pts_ICS = np.vstack([self.pts_ICS,point_ICS])
                        except:
                            self.pts_ICS = np.array(point_ICS,ndmin=2)
                        ind_del.append(i)
                    i += 1
                if len(ind_del)>0:
                    muscle = np.delete(muscle,ind_del,axis=0)
                try:
                    self.pts_ICS
                except:
                    self.pts_ICS = np.array([0,0],ndmin=2)

            #Get muscle points present in the current image (where Z~=0)                        
            for point in shifted_current:
                point = np.append(point,[1])
                point_ICS = np.delete(np.matmul(TIR,point),3)
                if abs(point_ICS[2])<0.1: 
                    point_ICS = np.delete(point_ICS,2)/0.0666666667
                    try:
                        self.current_pts_ICS = np.vstack([self.current_pts_ICS,point_ICS])
                    except:
                        self.current_pts_ICS = np.array(point_ICS,ndmin=2)    

            #If no muscle points in current image, set = [0,0]
            try:
                self.current_pts_ICS
            except:
                self.current_pts_ICS = np.array([0,0],ndmin=2)

        def get_volar_pts(self):
            if muscle == 'APB':
                self.get_muscle_surf()
            self.crop_image()

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
                    pixel_coord = pd.DataFrame([i,j],columns=['X','Y'])
                    if j<80:
                        j+=1
                        continue
                    if pixel == True and bright_found == False:
                        #only add to skin_surf if j is within 10 pixels of last j (skips bright particles/noise above skin surface)
                        if self.skin_surf.shape[0]>0:
                            if abs(self.skin_surf[-1,1]-j)<20:
                                self.skin_surf._append(pixel_coord,ignore_index=True)
                                self.skin_pts_RCS._append(ICS_to_RCS([i,j],self.TRI))
                                bright_found=True
                        else:
                            self.skin_surf._append(pixel_coordignore_index=True)
                            self.skin_pts_RCS._append(ICS_to_RCS([i,j],self.TRI))
                            bright_found=True

            try:
                #get max skin surface point
                skin_peakidx = pd.idxmin(self.skin_surf['Y'])
                self.skin_peak = self.skin_surf.iloc(skin_peakidx)
            except:
                self.skin_surf, self.skin_pts_RCS = np.array([0,0],ndmin=2), np.array([0,0,0],ndmin=2)
        
        #Get the volar muscle border points
        def get_muscle_surf(self):
            #apply hysteresis threshold
            hyst = self.apply_hyst_thresh()
            hyst_T = np.transpose(hyst)

            #Go along the skin surface and search for the muscle surface below it
            for point in self.skin_surf:
                j=point[0]
                i=point[1]
                for pixel in hyst_T[j,point[1]:]:
                    if pixel == False:
                        pixel_coord = [j,i]
                        try:
                            l2d_edge = np.vstack([l2d_edge,pixel_coord])
                        except:
                            l2d_edge = np.array(pixel_coord,ndmin=2)
                        break
                    i+=1

            #Apply canny edge detection
            canny = feature.canny(self.img,5.5,low_threshold = 0.05, high_threshold = 0.1)

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
                            volar_point = pd.DataFrame([point[0],i],columns=['X','Y'])
                            break
                    i+=-1
                    if j < lower_bound:
                        pixel = canny[j,point[0]]
                        if pixel == True:
                            volar_point = pd.DataFrame([point[0],j],columns=['X','Y'])
                            break
                    j+=1
                
                #If border point found, add it to volar_pts
                if pixel == True:
                    self.volar_pts._append(volar_point)    


            self.muscle_f = np.polyfit(self.volar_pts[:,0],self.volar_pts[:,1],deg = 2)



            #take curve fit to volar points of previous image. If volar points of current image aren't within 10 pixels of fitting curve, remove.
            if self.images.index(self)>self.seg_range[0]:
                self.current_img_index = self.images.index(self)
                f = self.images[self.current_img_index-1].muscle_f
            else: 
                f=self.muscle_f
            yi = np.polyval(f,self.volar_pts[:,0])
            i=0
            inidices_to_del = []
            for y in self.volar_pts[:,1]:
                if abs(y-yi[i])>45:
                    inidices_to_del.append(i)
                i+=1
            self.volar_pts = np.delete(self.volar_pts,inidices_to_del,axis=0)

        def get_dorsal_pts(self):
            #set img to cropped img and adjust volar points to fit cropped image
            #APB_points = self.pts_ICS - [self.x_adj,self.y_adj]
            adjust=[self.x_adj,self.y_adj]
            
            if muscle == 'APB':
                volar_pts = self.volar_pts - [self.x_adj,self.y_adj]

                #Interpolate cutoff points 
                f = scipy.interpolate.interp1d(volar_pts[:,0],volar_pts[:,1])
                xi = np.arange(np.min(volar_pts[:,0]),np.max(volar_pts[:,0]),1)
                yi = f(xi)
                interp_vp = np.stack([xi,yi],axis=1)

            img = self.cropped_img

            #Apply Canny Edge filter
            edges = feature.canny(img,CANNY_sigma,low_threshold = CANNY_lowt, high_threshold = CANNY_hight)
              
            #get points near muscle boundary, if first image: manual selection; otherwise, use previous b2mpeak
            #if manual selection, cutoff is set to volar points. If using previous b2mpeak, cutoff is set to previously selected points
            if self.images.index(self)==self.seg_range[0] or self.redo == True:
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

                edge_coords = array_to_xy(edges,True)

                #Find closest edge points to each clicked point
                #calculate distance between every clicked point and every edge coord, and find edge closest to clicked point
                closest_edges = pd.DataFrame(columns = ['X','Y'])
                img_borders=[]
                #Convert binary Canny Edge image to grayscale image
                edges_remember = binary_to_gs(edges)
                for n,point in enumerate(clicked_points):
                    edges_gs = binary_to_gs(edges)
                    closest_edges._append(find_closest_edge(point,edges))
                    edge = find_closest_edge(point,edges)
                    if edges_remember[edge[1],edge[0]] == 75:
                        i-=1
                        continue
                    edges_remember = flood_fill(edges_remember,(edge[1],edge[0]),75)
                    edges_gs = flood_fill(edges_gs,(edge[1],edge[0]),75)
                    img_borders.append(get_border(edges_gs,side))
                    edge_pts = array_to_xy(edges_gs,75)
                    edge_pts['edge_n'] = n
                    self.dorsal_pts._append(edge_pts)

                if closest_edges.shape[0]<1:
                    print("redo: no close edges found.")
                    self.redo = True
                    return

            else:
                adjust = [self.x_adj,self.y_adj]
                clicked_points = self.images[self.current_img_index-1].b2mpeak - adjust
                cutoff = self.images[self.current_img_index-1].b2mpeak - adjust
                previous_edges = self.images[self.current_img_index-1].b2mpeak_sep
                for edge in previous_edges:
                    edge -= adjust        

                #Interpolate cutoff points 
                f = scipy.interpolate.interp1d(cutoff[:,0],cutoff[:,1])
                xi = np.arange(np.min(cutoff[:,0]+1),np.max(cutoff[:,0])-4)
                yi = f(xi)
                interp_cutoff = np.stack([xi,yi],axis=1)     

                #Remove points outside of cutoff + cutoff radius  
                max_cutoff = int(np.max(interp_cutoff[:,1]))
                min_cutoff = int(np.min(interp_cutoff[:,1]))
                edges[0:min_cutoff-30,:] = False
                edges[max_cutoff+30:,:] = False
                for point in interp_cutoff:
                    if point[0]>edges.shape[1]-1 or point[0]<0: 
                        continue
                    for i in range(0,edges.shape[0]-1):
                        if i>point[1] + cutoff_radius or i < point[1] - cutoff_radius:
                            edges[i,int(point[0])]=False

                edge_coords = array_to_xy(edges,True)

                #Find closest edge points to each clicked point
                #calculate distance between every clicked point and every edge coord, and find edge closest to clicked point
                ind = np.lexsort((edge_coords[:,1],edge_coords[:,0]))
                edge_coords = edge_coords[ind]
                single_edge = np.array(edge_coords[0],ndmin=2)
                edge_coords = np.delete(edge_coords,0,axis=0)
                current_edges = []

                #group and sort edges of current image
                j=0
                while edge_coords.shape[0]>1:
                    while abs(single_edge[-1,0]-edge_coords[j,0])<=1:
                        diff = np.linalg.norm(single_edge[-1]-edge_coords[j])
                        if diff < 2:
                            single_edge = np.vstack([single_edge,edge_coords[j]])
                            edge_coords = np.delete(edge_coords,j,axis=0)
                        else:
                            j += 1
                        if j>=edge_coords.shape[0]-1: 
                            current_edges.append(single_edge)
                            single_edge = np.array(edge_coords[0],ndmin=2)
                            edge_coords = np.delete(edge_coords,0,axis=0)
                            j=0
                            break              
                    current_edges.append(single_edge)
                    if edge_coords.shape[0]<1:
                        break
                    single_edge = np.array(edge_coords[0],ndmin=2)
                    edge_coords = np.delete(edge_coords,0,axis=0)
                    j = 0
                    continue
                
                #Find closest edge to coordinates from previous edge
                close_edges = []
                for prev_edge in previous_edges:
                    min_x,max_x,min_y,max_y = np.min(prev_edge[:,0]),np.max(prev_edge[:,0]),np.min(prev_edge[:,1]),np.max(prev_edge[:,1])
                    dist_to_cur_edge = []
                    cur_edges = []
                    for cur_edge in current_edges:
                        x_min, y_min, x_max, y_max = np.min(cur_edge[:,0]), np.min(cur_edge[:,1]), np.max(cur_edge[:,0]), np.max(cur_edge[:,1])
                        if x_min < max_x and x_max > min_x and y_min < max_y+15 and y_max > min_y-15:
                            min_dists = []
                            for p in prev_edge:
                                edge_len = prev_edge.shape[0]
                                dist_to_p = []
                                for c in cur_edge:
                                    distance = np.linalg.norm(p-c)
                                    dist_to_p.append(distance)
                                if len(dist_to_p) > edge_len:
                                    dist_to_p.sort()
                                    dist_to_p = dist_to_p[0:edge_len]
                                min_dists.append(min(dist_to_p))
                            dist_to_cur_edge.append(sum(min_dists)/len(min_dists))
                            cur_edges.append(cur_edge)
                        
                    if len(dist_to_cur_edge)<1:
                        continue
                    closest_edge = cur_edges[dist_to_cur_edge.index(min(dist_to_cur_edge))]
                    close_edges.append(closest_edge)
                if len(close_edges)<1:
                    print("redo: no close edges found.2",previous_edges,current_edges)
                    self.redo = True
                    return
                
                self.b2mpeak_sep = []
                for edge in close_edges:
                    try:
                        self.dorsal_pts = np.vstack([self.dorsal_pts,edge])
                    except:
                        self.dorsal_pts = np.array(edge,ndmin=2)
                    edgesep = edge + adjust
                    self.b2mpeak_sep.append(edgesep)

            
            dmin = self.dorsal_pts[self.dorsal_pts.idxmin(self.dorsal_pts['Y'])]
            dmax = self.dorsal_pts[self.dorsal_pts.idxmax(self.dorsal_pts['Y'])]
            dminmax=[dmin,dmax]
            try:
                vmin = interp_vp[np.where(abs(interp_vp[:,1]-dmin[1])<10)]
                vmin = vmin[np.where(vmin[:,0] == np.min(vmin[:,0]))][0]
            except:
                vmin = [0]
            try:
                vmax = interp_vp[np.where(abs(interp_vp[:,1] - dmax[1])<10)]
                vmax = vmax[np.where(vmax[:,0] == np.max(vmax[:,0]))][0]
            except:
                vmax = [0]
            vminmax=[vmin,vmax]
            #close_edges[np.where(close_edges[:,0,:]==np.min(close_edges[:,0,:]))]
            emins,emaxes=[],[]
            for edge_n in range(0,n+1):
                emin,emax = np.min(edge[:,0]),np.max(edge[:,0])
                emins.append(emin)
                emaxes.append(emax)
            edge0 = close_edges[emins.index(min(emins))]
            edge1 = close_edges[emaxes.index(max(emaxes))]
            edges=[edge0,edge1]
            for i in range(0,2):
                edge = edges[i]
                if np.linalg.norm(dminmax[i][0]-vminmax[i][0])>130 or vminmax[i][0] == 0 or edge.shape[0]<10:
                    continue
                f = np.polyfit(edge[:,0],edge[:,1],deg = 2)
                if i == 0:
                    x=np.arange(vmin[0]-10,edge[0,0],1)
                else:
                    x=np.arange(edge[-1,0],vmax[0]+10,1)
                y = np.polyval(f,x)
                xy = np.stack((x,y),axis=-1)


            #Adjust extracted dorsal points to Image Coordinate System
            self.dorsal_pts[:,0] += self.x_adj
            self.dorsal_pts[:,1] += self.y_adj

            self.b2mpeak = self.dorsal_pts
            self.redo = False

        def crop_image(self):
            ymin, ymax = self.skin_peak[1], self.skin_peak[1]+350 
            self.cropped_img = self.img[ymin:ymax,:]
            
            sigma=5
            #blurred = skimage.filters.gaussian(self.cropped_img, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
            blurred = filters.apply_hysteresis_threshold(self.cropped_img,0.4,0.75)
            blurred = binary_to_gs(blurred)
            min_point = np.where(blurred[:,self.skin_peak[0]]==np.min(blurred[5:,self.skin_peak[0]]))[0][0]
            blurred=blurred*255
            flood = flood_fill(blurred,(min_point,self.skin_peak[0]),255,tolerance=100-blurred[min_point,self.skin_peak[0]])
            skin_flood = flood_fill(blurred,(self.skin_peak[1]-ymin,self.skin_peak[0]+2),255,tolerance=35)

            muscle_coords = np.where(flood == 255)
            skin_coords = np.where(skin_flood == 255)
            muscle_tip = np.min(muscle_coords[1])

            xmin, xmax = muscle_tip-10, np.max(self.skin_surf[:,0]-30)
            if self.n == 2 or self.n==3:
                xmax = np.max(self.skin_surf[:,0]-60)
            if self.n == 0 or self.n ==1:
                xmin = 0
            if self.n == 0:
                xmax+=30
            if self.n == 2 and self.current_img_index < 40 and muscle == 'APB':
                xmin = 220
                ymax=ymin+250
            if self.n == 2 and self.current_img_index < 15 and muscle == 'APB':
                xmin = 270
                ymax=ymin+150
            if self.n == 3 and self.current_img_index < 15 and muscle == 'APB':
                xmin = 160
            if self.n == 3:
                xmax -= 10
            if xmin<0:
                xmin=0            

            #delete points from muscle and skin surf that are cropped out
            if muscle == 'APB':
                self.volar_pts = np.delete(self.volar_pts,np.where(self.volar_pts[:,1]>ymax),axis=0)
                self.volar_pts = np.delete(self.volar_pts,np.where(self.volar_pts[:,0]<xmin),axis=0)
                self.volar_pts = np.delete(self.volar_pts,np.where(self.volar_pts[:,0]>xmax),axis=0)
            self.skin_surf = np.delete(self.skin_surf,np.where(self.skin_surf[:,1]>ymax),axis=0)
            self.skin_surf = np.delete(self.skin_surf,np.where(self.skin_surf[:,0]<xmin),axis=0)

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
            plt.scatter(self.dorsal_pts[:,0],self.dorsal_pts[:,1],color='blue',s=0.5)
            plt.scatter(self.pts_ICS[:,0],self.pts_ICS[:,1], color='green', s=0.3 )
            plt.scatter(self.current_pts_ICS[:,0],self.current_pts_ICS[:,1], color='purple', s=0.3 )
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


def get_border(img,side):
    #Get top points only
    flood_T = np.transpose(img)
    if side == 'dorsal':
        for i,col in enumerate(flood_T):
            top_found = False
            lower_pixel_searched = False
            for j,pixel in enumerate(col):
                if top_found == True and lower_pixel_searched == True:
                    flood[j,i] = 0
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
                    flood[j,i] = 0
                if top_found == True:
                    lower_pixel_searched = True
                if pixel == 75:
                    top_found = True
                j-=1
            i+=1
    
    return flood

def find_closest_edge(point,edges):
    for i in range(0,15):
        for j in range(-i-1,i+1):
            for k in [-i-1,i+1]:
                if point == True:
                    return pd.DataFrame([k,j],columns=['X','Y'])
    return pd.DataFrame(columns=['X','Y'])



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
    xy_coords=pd.DataFrame(columns=['X','Y'])
    for i,row in enumerate(img):
        for j,pixel in enumerate(row):
            if pixel == val:
                xy = pd.DataFrame([j,i],columns=['X','Y'])
                xy_coords._append(xy)

    return xy_coords

def get_TRI_array(TRE_folder,TEI_filename):
    TEI = np.loadtxt(TEI_filename, dtype=float)
    TRI_list=[]
    for i in range(0,4):
        T0n_filename = 'C:\\Users\\jocel\\OneDrive\\Desktop\\Sub00000009\\Trial 3\\T0' + str(i) + '.txt'
        T0n = np.loadtxt(T0n_filename, dtype=float)
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


subject_path = 'C:\\Users\\jocel\\OneDrive\\Desktop\\20220607 Formal Data Collection\\Sub00000007\\Trial 2\\'
TEI_path = 'C:\\Users\\jocel\\OneDrive\\Desktop\\20220607 Formal Data Collection\\Sub00000007\\TEI.txt'
APB_path = subject_path + 'APB Dense Point Cloudz.txt'
FPB_path = subject_path + 'FPB Dense Point Cloudz.txt'
OPP_path = subject_path + 'OPP Dense Point Cloudz.txt'

#4d array of all TRIs
TRI_all = get_TRI_array(subject_path,TEI_path)
try:
    APB = np.loadtxt(APB_path)
except:
    APB = np.array([0,0,0],ndmin=2)
try:
    OPP = np.loadtxt(OPP_path)
except:
    OPP = np.array([0,0,0],ndmin=2)
try:
    FPB = np.loadtxt(FPB_path)
except:
    FPB = np.array([0,0,0],ndmin=2)

muscles = [APB,FPB,OPP]
s0=Scan('0',6,42,[20,35]) 
#s1=Scan('1',5,57,[10,55]) 
s2=Scan('2',0,62,[15,45])
#s3=Scan('3',19,62,[20,45]) 
scans = [s2,s0]
for scan in [s0,s2]:
    print(scan.scan_folder)
    scan.skin_reconstruct()
    print(scan.max_skin_point)
current_muscle = np.array([0,0,0],ndmin = 2)

for scan in scans:
    
    scan.shift = (scan.max_skin_point-s0.max_skin_point)
    scan.segment_images()
    scan.shift_points(s0.max_skin_point)
    np.savetxt(subject_path +str(scan.folder)+ '\\' + muscle + side + 'points.txt',scan.dorsal_points)
    if muscle == 'APB':
        np.savetxt(subject_path +str(scan.folder) + '\\APBVolarPoints.txt',scan.volar_points)
        try:
            volar_pts = np.vstack([volar_pts,scan.volar_points])
        except:
            volar_pts = scan.volar_points
    try:
        dorsal_pts = np.vstack([dorsal_pts,scan.dorsal_points])
    except:
        dorsal_pts = scan.dorsal_points
    current_muscle = dorsal_pts



if muscle == 'APB':
    np.savetxt(subject_path + 'APBvolarPoints.txt',volar_pts)
np.savetxt(subject_path + muscle + side + 'points.txt',dorsal_pts)



 