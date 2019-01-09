"""Module for image reading/writing based on ITK library

Functions:
    ImageReader: returns ITK image
    ImageWriter: writes image with the specified format
    StlReader: loads an STL 3D model    
"""

####  PYTHON and ITK/VTK MODULES
import itk
import numpy as np
import vtk



def ImageReader(image_file_path, image_type, compute_info = True):

    """Function to read an image (either fixed or moving) with itk and possibly retrieve its info (spacing, origin, size, volume centre)

        Args:
            image_file_path
            itk image type
            compute_info = if True computes spacing, origin, size, volume centre (True by default)

        Returns:
            image
            image_info as dictionary with Keys (Spacing, Origin, Size, Volume_center), if compute info
    """

    # create itk reader
    imageReader  = itk.ImageFileReader[image_type].New()
    imageReader.SetFileName( image_file_path )

    # Update reader
    imageReader.Update()
    image = imageReader.GetOutput()

    if compute_info:

        # Compute spacing, origin, size
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        size = image.GetBufferedRegion().GetSize()

        volume_center = np.asarray(origin) + np.multiply(spacing, np.divide(size,2.)) - np.divide(spacing,2.)

        image_info = {'Spacing': spacing, 'Origin': origin, 'Size': size, 'Volume_center': volume_center}

        return image, image_info

    else:

        return image 


def ImageWriter(image, image_type, image_file_name, extension = '.mha'):

    """Function to write an image (either fixed or moving) with itk

        Args:
            image_file_path
            itk image type
            image file extension
    """

    # create itk writee
    imageWriter = itk.ImageFileWriter[image_type].New()

    imageWriter.SetFileName( image_file_name + extension)
    imageWriter.SetInput( image )
    imageWriter.Update()


def StlReader(StlModelFileName, import_mode = 'vtk'):

    """Function to read an stl file and returns it as a numpy array

        Args:
            stl_file_path
            import_mode (so far only vtk)
    """

    if import_mode == 'vtk':

        # Load stl file
        readerSTL = vtk.vtkSTLReader()
        readerSTL.SetFileName(StlModelFileName)
        # 'update' the reader i.e. read the .stl file
        readerSTL.Update()

        polydata = readerSTL.GetOutput()

        # If there are no points in 'vtkPolyData' something went wrong
        if polydata.GetNumberOfPoints() == 0:
            raise ValueError(
                "No point data could be loaded from '" + filenameSTL)
            return None
        
        return polydata