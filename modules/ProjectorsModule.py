"""Module for generation of Digitally Reconstructed Radiographs (DRR).

This module includes classes for generation of DRRs from either a volumetric image (CT,MRI) 
or a STL model, and a projector class factory.

Classes:
    SiddonGpu: GPU accelerated (CUDA) DRR generation from CT or MRI scan.  
    Mahfouz: binary DRR generation from CAD model in STL format.

Functions:
    projector_factory: returns a projector instance.
    
New projectors can be plugged-in and added to the projector factory
as long as they are defined as classes with the following methods:
    compute: returns a 2D image (DRR) as a numpy array.
    delete: eventually deletes the projector object (only needed to deallocate memory from GPU) 
"""

####  PYTHON MODULES
import numpy as np
import time
import sys

####  Python ITK/VTK MODULES
import itk
import cv2
import vtk
from vtk.util import numpy_support



####  MY MODULES
import ReadWriteImageModule as rw
import RigidMotionModule as rm

sys.path.append('../wrapped_modules/')
from SiddonGpuPy import pySiddonGpu     # Python wrapped C library for GPU accelerated DRR generation



def projector_factory(projector_info,
                      movingImageFileName,
                      PixelType = itk.F,
                      Dimension = 3,
                      ScalarType = itk.D):

    """Generates instances of the specified projectors.

    Args:
        projector_info (dict of str): includes camera intrinsic parameters and projector-specific parameters
        movingImageFileName (string): cost function returning the metric value

    Returns:
        opt: instance of the specified projector class.
    """

    if projector_info['Name'] == 'SiddonGpu':

        p = SiddonGpu(projector_info,
                      movingImageFileName,
                      PixelType,
                      Dimension,
                      ScalarType)

        return p


    if projector_info['Name'] == 'Mahfouz':

        p = Mahfouz(projector_info,
                      movingImageFileName,
                      PixelType,
                      Dimension,
                      ScalarType)

        return p



class Mahfouz():

    """Binary DRR generation from STL model.

       This class renders a binary image using VTK and smooths it with Gaussian filter using OpenCV.

       Methods:
            _new_render (function): sets up a new renderer for a new DRR
            compute (function): returns a 2D image (DRR) as a numpy array.
            delete (function): eventually deletes the projector object (only needed to deallocate memory from GPU) 

       Note:the camera coordinate system has the y axis pointing downwards

    """


    def __init__(self,projector_info,
                      StlModelFileName,
                      PixelType,
                      Dimension,
                      ScalarType,
                      correction_matrix = []):

        """Prepares itk stuff, loads stl and initializes VTK renderer.

           Args:
                projector_info (dict of str): with the following keys:
                    - Name: 'Mahfouz'
                    - near (float): near clipping plane (vtk)
                    - far (float): far clipping plane (vtk)
                    - intGsigma (int): sigma of the Gaussian filter
                    - intGsize (int): size of the Gaussian kernel
                    - 3Dmodel (str):  name of 3D model (ie. stem)

           Useful links:
           http://ksimek.github.io/2013/08/13/intrinsic/
           http://www.songho.ca/opengl/gl_transform.html
           https://gist.github.com/benoitrosa/ffdb96eae376503dba5ee56f28fa0943
        """

        # ITK: Instantiate types
        self.Dimension = 2
        self.ImageType2D = itk.Image[PixelType, 2]
        self.RegionType = itk.ImageRegion[self.Dimension]
        movImageInfo = {'Volume_center': (0.0, 0.0)}
        self.correction_matrix = correction_matrix

        # ITK: Set DRR image at initial position (at +focal length along the z direction)
        DRR = self.ImageType2D.New()
        self.DRRregion = self.RegionType()
        self.movDirection = DRR.GetDirection()

        DRRstart = itk.Index[self.Dimension]()
        DRRstart.Fill(0)

        self.DRRsize = [0]*self.Dimension
        self.DRRsize[0] = projector_info['DRRsize_x']
        self.DRRsize[1] = projector_info['DRRsize_y']

        self.DRRregion.SetSize(self.DRRsize)
        self.DRRregion.SetIndex(DRRstart)

        self.DRRspacing = itk.Point[itk.F, self.Dimension]()
        self.DRRspacing[0] = projector_info['DRRspacing_x']
        self.DRRspacing[1] = projector_info['DRRspacing_y']

        self.DRRorigin = itk.Point[itk.F, self.Dimension]()
        self.DRRorigin[0] = movImageInfo['Volume_center'][0] - projector_info['DRR_ppx'] - self.DRRspacing[0]*(self.DRRsize[0] - 1.) / 2.
        self.DRRorigin[1] = movImageInfo['Volume_center'][1] - projector_info['DRR_ppy'] - self.DRRspacing[1]*(self.DRRsize[1] - 1.) / 2.

        DRR.SetRegions(self.DRRregion)
        DRR.Allocate()
        DRR.SetSpacing(self.DRRspacing)
        DRR.SetOrigin(self.DRRorigin)
        self.movDirection.SetIdentity()
        DRR.SetDirection(self.movDirection)

        # Load stl mesh (with vtk function)
        self.StlMesh = rw.StlReader(StlModelFileName)

        ## the correction matrix allows to make the local CS of the object coincide with the standard camera CS
        if self.correction_matrix:
            self.StlPoints = np.dot(correction_matrix, rm.augment_matrix_coord(StlPoints))[0:3].T

        # Set Camera parameters (convert from mm to pixel units)    
        self.ppx_pixels = projector_info['DRR_ppx']/self.DRRspacing[0]
        self.ppy_pixels = projector_info['DRR_ppy']/self.DRRspacing[1]
        self.focal_length_pixels = projector_info['focal_lenght']/self.DRRspacing[0]
        self.near = projector_info['near']
        self.far = projector_info['far']

        # Prepare Gaussian filters for intensity image
        self.IntGaussSigma = projector_info['intGsigma']
        self.IntGaussSize = projector_info['intGsize']

        # VTK: Initialize first rendering
        #Mapper
        init_mapper = vtk.vtkPolyDataMapper()
        init_mapper.SetInputData(self.StlMesh)

        # Actor for binary image
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(init_mapper)
        self.actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        self.actor.GetProperty().SetAmbient(1)
        self.actor.GetProperty().SetDiffuse(0)
        self.actor.GetProperty().SetSpecular(0)

        # Renderer for binary image
        self.MahfouzRenderer = vtk.vtkRenderer()
        self.MahfouzRenderer.AddActor(self.actor)
        self.MahfouzRenderer.SetBackground(1, 1, 1)

        # Renderer window
        self.MahfouzRenderWindow = vtk.vtkRenderWindow()
        self.MahfouzRenderWindow.AddRenderer(self.MahfouzRenderer)
        self.MahfouzRenderWindow.SetOffScreenRendering(1) # it prevents generating a window
        self.MahfouzRenderWindow.SetSize(self.DRRsize[0], self.DRRsize[1])
        self.MahfouzRenderWindow.Render()

        # Camera parameters
        self.Camera = self.MahfouzRenderer.GetActiveCamera()
        self.Camera.SetClippingRange(self.near, self.far)
        self.Camera.SetPosition(0, 0, 0)
        self.Camera.SetFocalPoint(0, 0, -1)
        self.Camera.SetViewUp(0, 1, 0)    
           
        # Set window center for offset principal point
        # if principal point is referred to principal ray
        wcx = -2.0*(self.ppx_pixels) /self.DRRsize[0]
        wcy = -2.0*(self.ppy_pixels ) /self.DRRsize[1]
        # if principal point is referred to image origin (bottom left)
        #wcx = -2.0*(self.ppx_pixels - self.DRRsize[0] / 2.0) / self.DRRsize[0]
        #wcy = 2.0*(self.ppy_pixels - self.DRRsize[1] / 2.0) / self.DRRsize[1]
        self.Camera.SetWindowCenter(wcx, wcy)        
        # Set vertical view angle as a indirect way of setting the y focal distance
        angle = 180.0 / np.pi * 2.0 * np.arctan2(self.DRRsize[1] / 2.0, self.focal_length_pixels)
        self.Camera.SetViewAngle(angle)

        # Render window
        self.MahfouzRenderWindow.Render()

        # Initial Window to image filter
        init_windowToImageFilter = vtk.vtkWindowToImageFilter()
        init_windowToImageFilter.SetInput(self.MahfouzRenderWindow)
        init_windowToImageFilter.Update()



    def _new_render(self, transform_parameters):

        """Sets a new VTK renderer for a new DRR.

           This function is called by the class method compute() 
        """

        # Get transform parameters
        rotx = transform_parameters[0]
        roty = transform_parameters[1]
        rotz = transform_parameters[2]
        tx = transform_parameters[3]
        ty = transform_parameters[4]
        tz = transform_parameters[5]

        # Create a copy of the mesh (done every time, otherwise I could not make it work)
        polydata = vtk.vtkPolyData()
        polydata.DeepCopy(self.StlMesh)

        # Get points of copied mesh
        to_transform = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

        # Get ZXY transformation matrix
        T = rm.get_rigid_motion_mat_from_euler(rotz, 'z', rotx, 'x', roty, 'y', tx, ty, tz)
        
        # Correct if needed
        if self.correction_matrix:
            T = np.dot(self.correction_matrix, T)

        # Apply rotation 
        # ACHTUNG: transpose of T needed if post-multiplication is needed.
        #          post-multiplication is needed to allow C-contiguity of points vector        
        transformed = np.dot(to_transform, T[0:3, 0:3].T)

        # Apply translation
        transformed += T[0:3,3]

        transformed_vtk = numpy_support.numpy_to_vtk(transformed) 
        points_vtk = vtk.vtkPoints()
        points_vtk.SetData(transformed_vtk)
        polydata.SetPoints(points_vtk)

        # Set new mapper
        new_mapper = vtk.vtkPolyDataMapper()
        new_mapper.SetInputData(polydata)

        # Remove old actor
        self.MahfouzRenderer.RemoveActor(self.actor)

        # Set new actor (for binary image): it overwrites self.actor!
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(new_mapper)
        self.actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        self.actor.GetProperty().SetAmbient(1)
        self.actor.GetProperty().SetDiffuse(0)
        self.actor.GetProperty().SetSpecular(0)

        # Add new actor        
        self.MahfouzRenderer.AddActor(self.actor)            

        # New Window to image filter
        new_windowToImageFilter = vtk.vtkWindowToImageFilter()
        new_windowToImageFilter.SetInput(self.MahfouzRenderWindow)
        new_windowToImageFilter.Update()

        ## Write init
        #writer = vtk.vtkPNGWriter()
        #writer.SetFileName('BinaryVTK.png')
        #writer.SetInputConnection(new_windowToImageFilter.GetOutputPort())
        #writer.Write()

        # Get numpy array out of final vtk rendered image (uint8)
        DRRvtk = new_windowToImageFilter.GetOutput()
        DRRvtk_array = DRRvtk.GetPointData().GetScalars()
        DRRvtk_np_array = numpy_support.vtk_to_numpy(DRRvtk_array)

        # Extract one component (any, they are all the same) and reshape
        DRRvtk_np_array_1d = DRRvtk_np_array[:,0]
        outputDRR = np.reshape(DRRvtk_np_array_1d, (self.DRRsize[0], self.DRRsize[1]))

        return outputDRR


    def compute(self, transform_parameters):

        """Generates a DRR given the transform parameters.

           Args:
               transform_parameters (list of floats): rotX, rotY,rotZ, transX, transY, transZ
 
        """

        # Instantiate 3D DRR image at its initial position (at +focal length along the z direction)
        newDRR = self.ImageType2D.New()

        newDRR.SetRegions(self.DRRregion)
        newDRR.Allocate()
        newDRR.SetSpacing(self.DRRspacing)
        newDRR.SetOrigin(self.DRRorigin)
        self.movDirection.SetIdentity()
        newDRR.SetDirection(self.movDirection)

        # Get 3d array for DRR (where to store the final output, in the image plane that in fact does not move)
        #newDRRArray = itk.PyBuffer[self.ImageType].GetArrayFromImage(newDRR)
        newDRRArray = itk.GetArrayViewFromImage(newDRR)

        # Render binary image   
        renderedDRR = self._new_render(transform_parameters)

        # perform gaussian blur -> final intensity image
        blurredDRR = cv2.GaussianBlur( renderedDRR, self.IntGaussSize, self.IntGaussSigma)

        # flip
        blurredDRR = cv2.flip(blurredDRR, 0)

        # Re-copy into original image array, hence into original image (since the former is just a view of the latter)
        newDRRArray.setfield(blurredDRR,newDRRArray.dtype)

        # Update itk image
        newDRR.UpdateOutputInformation()

        return newDRR


    def delete(self):
        
        """Not needed to deallocate anything"""

        return



class SiddonGpu():

    """GPU accelearated DRR generation from volumetric image (CT or MRI scan).

       This class renders a DRR from a volumetric image, with an accelerated GPU algorithm
       from a Python wrapped library (SiddonGpuPy), written in C++ and accelerated with Cuda.
       IMplementation is based both on the description found in the “Improved Algorithm” section in Jacob’s paper (1998): 
       https://www.researchgate.net/publication/2344985_A_Fast_Algorithm_to_Calculate_the_Exact_Radiological_Path_Through_a_Pixel_Or_Voxel_Space
       and on the implementation suggested in Greef et al 2009:
       https://www.ncbi.nlm.nih.gov/pubmed/19810482

       Methods:
            compute (function): returns a 2D image (DRR) as a numpy array.
            delete (function): deletes the projector object (needed to deallocate memory from GPU)
    """


    def __init__(self, projector_info,
                       movingImageFileName,
                       PixelType,
                       Dimension,
                       ScalarType):

        """Reads the moving image and creates a siddon projector 
           based on the camera parameters provided in projector_info (dict)
        """

        # ITK: Instantiate types
        self.Dimension = Dimension
        self.ImageType = itk.Image[PixelType, Dimension]
        self.ImageType2D = itk.Image[PixelType, 2]
        self.RegionType = itk.ImageRegion[Dimension]
        PhyImageType=itk.Image[itk.Vector[itk.F,Dimension],Dimension] # image of physical coordinates

        # Read moving image (CT or MRI scan)
        movImage, movImageInfo = rw.ImageReader(movingImageFileName, self.ImageType)
        self.movDirection = movImage.GetDirection()

        # Calculate side planes
        X0 = movImageInfo['Volume_center'][0] - movImageInfo['Spacing'][0]*movImageInfo['Size'][0]*0.5
        Y0 = movImageInfo['Volume_center'][1] - movImageInfo['Spacing'][1]*movImageInfo['Size'][1]/2.0
        Z0 = movImageInfo['Volume_center'][2] - movImageInfo['Spacing'][2]*movImageInfo['Size'][2]/2.0

        # Get 1d array for moving image
        #movImgArray_1d = np.ravel(itk.PyBuffer[self.ImageType].GetArrayFromImage(movImage), order='C') # ravel does not generate a copy of the array (it is faster than flatten)
        movImgArray_1d = np.ravel(itk.GetArrayFromImage(movImage), order='C') # ravel does not generate a copy of the array (it is faster than flatten)

        # Set parameters for GPU library SiddonGpuPy
        NumThreadsPerBlock = np.array( [projector_info['threadsPerBlock_x'], projector_info['threadsPerBlock_y'], projector_info['threadsPerBlock_z'] ] )
        DRRsize_forGpu = np.array([ projector_info['DRRsize_x'], projector_info['DRRsize_y'], 1 ])
        MovSize_forGpu = np.array([ movImageInfo['Size'][0], movImageInfo['Size'][1], movImageInfo['Size'][2] ])
        MovSpacing_forGpu = np.array([ movImageInfo['Spacing'][0], movImageInfo['Spacing'][1], movImageInfo['Spacing'][2] ]).astype(np.float32)

        # Define source point at its initial position (at the origin = moving image center)
        self.source = [0]*Dimension
        self.source[0] = movImageInfo['Volume_center'][0]
        self.source[1] = movImageInfo['Volume_center'][1]
        self.source[2] = movImageInfo['Volume_center'][2]

        # Set DRR image at initial position (at +focal length along the z direction)
        DRR = self.ImageType.New()
        self.DRRregion = self.RegionType()

        DRRstart = itk.Index[Dimension]()
        DRRstart.Fill(0)

        self.DRRsize = [0]*Dimension
        self.DRRsize[0] = projector_info['DRRsize_x']
        self.DRRsize[1] = projector_info['DRRsize_y']
        self.DRRsize[2] = 1

        self.DRRregion.SetSize(self.DRRsize)
        self.DRRregion.SetIndex(DRRstart)

        self.DRRspacing = itk.Point[itk.F, Dimension]()
        self.DRRspacing[0] = projector_info['DRRspacing_x']
        self.DRRspacing[1] = projector_info['DRRspacing_y']
        self.DRRspacing[2] = 1.     

        self.DRRorigin = itk.Point[itk.F, Dimension]()
        self.DRRorigin[0] = movImageInfo['Volume_center'][0] - projector_info['DRR_ppx'] - self.DRRspacing[0]*(self.DRRsize[0] - 1.) / 2.
        self.DRRorigin[1] = movImageInfo['Volume_center'][1] - projector_info['DRR_ppy'] - self.DRRspacing[1]*(self.DRRsize[1] - 1.) / 2.
        self.DRRorigin[2] = movImageInfo['Volume_center'][2] + projector_info['focal_lenght'] #/ 2.

        DRR.SetRegions(self.DRRregion)
        DRR.Allocate()
        DRR.SetSpacing(self.DRRspacing)
        DRR.SetOrigin(self.DRRorigin)
        self.movDirection.SetIdentity()
        DRR.SetDirection(self.movDirection)

        # Get array of physical coordinates for the DRR at the initial position 
        PhysicalPointImagefilter=itk.PhysicalPointImageSource[PhyImageType].New()
        PhysicalPointImagefilter.SetReferenceImage(DRR)
        PhysicalPointImagefilter.SetUseReferenceImage(True)
        PhysicalPointImagefilter.Update()
        sourceDRR = PhysicalPointImagefilter.GetOutput()

        #self.sourceDRR_array_to_reshape = itk.PyBuffer[PhyImageType].GetArrayFromImage(sourceDRR)[0].copy(order = 'C') # array has to be reshaped for matrix multiplication
        self.sourceDRR_array_to_reshape = itk.GetArrayFromImage(sourceDRR)[0] # array has to be reshaped for matrix multiplication

        tGpu1 = time.time()

        # Generate projector object
        self.projector = pySiddonGpu(NumThreadsPerBlock,
                                  movImgArray_1d,
                                  MovSize_forGpu,
                                  MovSpacing_forGpu,
                                  X0.astype(np.float32), Y0.astype(np.float32), Z0.astype(np.float32),
                                  DRRsize_forGpu)

        tGpu2 = time.time()

        print( '\nSiddon object initialized. Time elapsed for initialization: ', tGpu2 - tGpu1, '\n')



    def compute(self, transform_parameters):


        """Generates a DRR given the transform parameters.

           Args:
               transform_parameters (list of floats): rotX, rotY,rotZ, transX, transY, transZ
 
        """

        #tDRR1 = time.time()

        # Get transform parameters
        rotx = transform_parameters[0]
        roty = transform_parameters[1]
        rotz = transform_parameters[2]
        tx = transform_parameters[3]
        ty = transform_parameters[4]
        tz = transform_parameters[5]

        # compute the transformation matrix and its inverse (itk always needs the inverse)
        Tr = rm.get_rigid_motion_mat_from_euler(rotz, 'z', rotx, 'x', roty, 'y', tx, ty, tz)
        invT = np.linalg.inv(Tr) # very important conversion to float32, otherwise the code crashes

        # Move source point with transformation matrix
        source_transformed = np.dot(invT, np.array([self.source[0],self.source[1],self.source[2], 1.]).T)[0:3]
        source_forGpu = np.array([ source_transformed[0], source_transformed[1], source_transformed[2] ], dtype=np.float32)

        # Instantiate new 3D DRR image at its initial position (at +focal length along the z direction)
        newDRR = self.ImageType.New()

        newDRR.SetRegions(self.DRRregion)
        newDRR.Allocate()
        newDRR.SetSpacing(self.DRRspacing)
        newDRR.SetOrigin(self.DRRorigin)
        self.movDirection.SetIdentity()
        newDRR.SetDirection(self.movDirection)

        # Get 3d array for DRR (where to store the final output, in the image plane that in fact does not move)
        #newDRRArray = itk.PyBuffer[self.ImageType].GetArrayFromImage(newDRR)
        newDRRArray = itk.GetArrayViewFromImage(newDRR)

        #tDRR3 = time.time()

        # Get array of physical coordinates of the transformed DRR
        sourceDRR_array_reshaped = self.sourceDRR_array_to_reshape.reshape((self.DRRsize[0]*self.DRRsize[1], self.Dimension), order = 'C')

        sourceDRR_array_transformed = np.dot(invT, rm.augment_matrix_coord(sourceDRR_array_reshaped))[0:3].T # apply inverse transform to detector plane, augmentation is needed for multiplication with rigid motion matrix

        sourceDRR_array_transf_to_ravel = sourceDRR_array_transformed.reshape((self.DRRsize[0],self.DRRsize[1], self.Dimension), order = 'C')

        DRRPhy_array = np.ravel(sourceDRR_array_transf_to_ravel, order = 'C').astype(np.float32)

        # Generate DRR
        #tGpu3 = time.time()
        output = self.projector.generateDRR(source_forGpu,DRRPhy_array)
        #tGpu4 = time.time()

        # Reshape copy
        #output_reshaped = np.reshape(output, (self.DRRsize[2], self.DRRsize[1], self.DRRsize[0]), order='C') # no guarantee about memory contiguity
        output_reshaped = np.reshape(output, (self.DRRsize[1], self.DRRsize[0]), order='C') # no guarantee about memory contiguity

        # Re-copy into original image array, hence into original image (since the former is just a view of the latter)
        newDRRArray.setfield(output_reshaped,newDRRArray.dtype)

        # Redim filter to convert the DRR from 3D slice to 2D image (necessary for further metric comparison)
        filterRedim = itk.ExtractImageFilter[self.ImageType, self.ImageType2D].New()
        filterRedim.InPlaceOn()
        filterRedim.SetDirectionCollapseToSubmatrix()

        newDRR.UpdateOutputInformation() # important, otherwise the following filterRayCast.GetOutput().GetLargestPossibleRegion() returns an empty image

        size_input = newDRR.GetLargestPossibleRegion().GetSize()
        start_input = newDRR.GetLargestPossibleRegion().GetIndex()

        size_output = [0]*self.Dimension
        size_output[0] = size_input[0]
        size_output[1] = size_input[1]
        size_output[2] = 0

        sliceNumber = 0
        start_output = [0]*self.Dimension
        start_output[0] = start_input[0]
        start_output[1] = start_input[1]
        start_output[2] = sliceNumber

        desiredRegion = self.RegionType()
        desiredRegion.SetSize( size_output )
        desiredRegion.SetIndex( start_output )

        filterRedim.SetExtractionRegion( desiredRegion )

        filterRedim.SetInput(newDRR)

        #tDRR2 = time.time()

        filterRedim.Update()

        #print( '\nTime elapsed for generation of DRR: ', tDRR2 - tDRR1)

        return filterRedim.GetOutput()
        


    def delete(self):
        
        """Deletes the projector object >>> GPU is freed <<<"""

        self.projector.delete()