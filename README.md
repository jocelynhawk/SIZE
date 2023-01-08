SIZE
========

SIZE develops 3D models of the thenar muscles from ultrasound images, and performs geometric calculations of those models. Robot-assisted ultrasound was used to collect images and provide probe position information for each image. 

*ultrasound data needed to run 20220524ImageSegmentation.py is not provided due to human subject privacy

Features
--------

- Image segmentation
- GUI programming
- 3D mesh processing
- Coordinate system transformations

Dependencies
--------
-Tkinter
-NumPy
-Vtk
-PyVista
-Matplotlib
-SciPy

Usage
--------

20220524ImageSegmentation.py semi-automatically segments the ultrasound images to create a 3D point cloud of the targeted muscle. The user manually selects one or more points near targeted muscle borders from the first ultrasound image, and the program extracts the muscle borders using thresholding and canny-edge detection. The program then automatically segments the succeeding images by tracking the extracted muscle borders from the first image. If they cannot be found, the program prompts the user to select new points. Extracted points are transformed from the 2D image coordinate system to the 3D robot coordinate system using a series of transformation matrices.

MeshReconstruction.py creates a 3D triangulated mesh from the 3D point cloud of each muscle.

MeshCalculations.py calculates muscle length, volume, and nominal cross-sectional area from the 3D meshes.