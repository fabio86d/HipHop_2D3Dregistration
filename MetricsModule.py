"""Module including similarity metrics for 2D/2D image registration.

This module includes a colleciton of classes for evaluation of the similarity between two 2D images,
and a metric class factory.

Classes:
    MattesMutualv4_metric: Mutual information metric (ITK)
    MattesMutualv3_metric: Mutual information metric (ITK)
    ViolaMutual_metric: Mutual information metric (ITK)
    MeanSquares_metric: Mean squared differences metric (ITK)
    NormCorr_metric: Normalized cross-correlation metric (ITK)
    MeanRec_metric: Mean reciprocal squared differences (ITK)
    GradCorr_metric: Gradient Correlation metric
    Mahfouz_metric: Custom metric based on 
        "A robust method for registration of three-dimensional knee implant models 
        to two-dimensional fluoroscopy images", Mahfouz et al. 2003, IEEE Transactions on Medical Imaging
    MahfouzHipHop_metric

Functions:
    metric_factory: returns a metric instance.
    
New metrics can be plugged-in and added to the metric factory
as long as they are defined as classes with the following methods:
    update: sets new fixed image and new mask image
    compute: returns similarity measure between (masked) fixed image
             and current (masked) moving image.
"""


####  PYTHON MODULES
import numpy as np
import time


####  Python ITK/VTK/OpenCV MODULES
import itk
import vtk
from vtk.util import numpy_support
import cv2



def metrics_factory(metric_info, 
                     PixelType = itk.F,
                     Dimension = 2,
                     ScalarType = itk.D):

    """Generates instances of the specified metrics.

    Args:
        metric_info (dict of str): includes metric-specific parameters

    Returns:
        opt: instance of the specified metric class.
    """

    if metric_info['Name'] == 'MattesMutualv4':

        m= MattesMutualv4_metric(metric_info, PixelType, Dimension, ScalarType)

        return m

    if metric_info['Name'] == 'MattesMutualv3':

        m= MattesMutualv3_metric(metric_info, PixelType, Dimension, ScalarType)

        return m

    if metric_info['Name'] == 'ViolaMutual':

        m= ViolaMutual_metric(metric_info, PixelType, Dimension, ScalarType)

        return m

    if metric_info['Name'] == 'MeanSquares':

        m= MeanSquares_metric(metric_info, PixelType, Dimension, ScalarType)

        return m

    if metric_info['Name'] == 'NormCorr':

        m= NormCorr_metric(metric_info, PixelType, Dimension, ScalarType)

        return m

    if metric_info['Name'] == 'MeanRec':

        m= MeanRec_metric(metric_info, PixelType, Dimension, ScalarType)

        return m

    if metric_info['Name'] == 'GradCorr':

        m= GradCorr_metric(metric_info, PixelType, Dimension, ScalarType)

        return m

    if metric_info['Name'] == 'Mahfouz':

        m= Mahfouz_metric(metric_info, PixelType, Dimension, ScalarType)

        return m

    if metric_info['Name'] == 'MahfouzHipHop':

        m= MahfouzHipHop_metric(metric_info, PixelType, Dimension, ScalarType)

        return m

    return


class Mahfouz_metric():

    """Custom metric inspired by the work from Mahfouz et al.

       "A robust method for registration of three-dimensional knee implant models 
       to two-dimensional fluoroscopy images", Mahfouz et al. 2003, IEEE Transactions on Medical Imaging 
    """

    def __init__(self, metric_info, 
                       PixelType = itk.F,
                       Dimension = 2,
                       ScalarType = itk.D):

        """Based on metric_info (dict), sets up the parameters for Canny Edge extraction and Gaussian Blur."""

        self.info = metric_info
        self.FixedImg = []
        self.MaskImg = []

        # Prepare Gaussian filter for edge image
        self.EdgeGaussSigma = metric_info['edgeGsigma']
        self.EdgeGaussSize = metric_info['edgeGsize']

        # Prepare Canny edge detector for both Moving and Fixed image
        self.MovCannyThr1 = metric_info['MovCannyThr1']
        self.MovCannyThr2 = metric_info['MovCannyThr2']
        self.FixCannyThr1 = metric_info['FixCannyThr1']
        self.FixCannyThr2 = metric_info['FixCannyThr2']
        self.FixCannyAperture = metric_info['FixCannyAperture']

        # Mahfouz weights
        self.intWeight = metric_info['intWeight']
        self.edgeWeight = metric_info['edgeWeight']

        self.saveImages = metric_info['saveImages']

        self.temp_counter_fix = 0
        self.temp_counter_mov = 0


    def update(self, itkFixedImage, maskImage, current_settings = None):

        """Sets new fixed image and new mask image"""

        # Since the HipHop pipeline deals with itk images, itkFixedImage and maskImage
        # need to be converted to numpy arrays for the Mahfouz metric.
        
        # Get COPY OF numpy array of fixed image (not just GetArrayViewFromImage!)
        self.FixedImg = itk.GetArrayFromImage(itkFixedImage)

        # Rescale image to 8bit
        min_img_contrast = np.amin(self.FixedImg)
        max_img_contrast = np.amax(self.FixedImg)

        # rescale 16bit into 8bit with contrast stretching
        img_for_canny = self._look_up_table(self.FixedImg, 0, 65535)

        # generate canny edge fixed image
        #FixedEdgeImg = cv2.Canny(self.FixedImg.astype(np.uint8), self.FixCannyThr1, self.FixCannyThr2).astype(np.float64)  
        FixedEdgeImg = cv2.Canny(img_for_canny, self.FixCannyThr1, self.FixCannyThr2,L2gradient=False,apertureSize= self.FixCannyAperture)  

        # blurr edge image (rescaling to uint16 is necessary)
        self.FixedEdgeImg = cv2.GaussianBlur( FixedEdgeImg*((np.power(2,16) - 1)/(np.power(2,8) - 1)), self.EdgeGaussSize, self.EdgeGaussSigma)

        # At the moment the mask is ignored

        # Save images
        if self.saveImages:

            # Save fixed edge images
            cv2.imwrite('FixedEdgImg' + str(self.temp_counter_fix) + '.tif',self.FixedEdgeImg)
            cv2.imwrite('FixedImg' + str(self.temp_counter_fix) + '.tif',img_for_canny.astype(np.uint8))
            self.temp_counter_fix = self.temp_counter_fix + 1


    def compute(self, movImage):

        """Returns similarity measure between (masked) fixed image and current (masked) moving image."""

        # Get COPY OF numpy array of moving image (not just GetArrayViewFromImage!)
        movImg = itk.GetArrayFromImage(movImage)

        # generate canny edge moving image
        movEdgeImg = cv2.Canny(movImg.astype(np.uint8), self.MovCannyThr1, self.MovCannyThr2).astype(np.float64) 

        # blurr edge image (rescaling to uint16 is necessary)
        movEdgeImg = cv2.GaussianBlur( movEdgeImg*((np.power(2.,16.) - 1.)/(np.power(2.,8.) - 1.)) , self.EdgeGaussSize, self.EdgeGaussSigma)

        if self.saveImages:

            # Save fixed edge images
            cv2.imwrite('MovEdgImg' + str(self.temp_counter_mov) + '.tif',movEdgeImg)
            cv2.imwrite('MovImg' + str(self.temp_counter_mov) + '.tif',movImg.astype(np.uint8))
            self.temp_counter_mov = self.temp_counter_mov + 1
        
        # Compute intensity fitness
        int_fitness = np.sum(np.multiply(self.FixedImg.astype(float),movImg.astype(float)))/np.sum(movImg.astype(float))

        # Compute edge fitness
        edge_fitness = np.sum(np.multiply(self.FixedEdgeImg.astype(float),movEdgeImg.astype(float)))/np.sum(movEdgeImg.astype(float))

        return -self.intWeight*int_fitness - self.edgeWeight*edge_fitness


    # LOOK UP TABLE: FASTER CONTRAST STRETCHING METHOD (from 16bit to 8bit ONLY)
    def _clip_and_rescale(self, img, min, max):

        image = np.array(img, copy = True) # just create a copy of the array
        image.clip(min,max, out = image)
        image -= min
        #image //= (max - min + 1)/256.
        image = np.divide(image,(max - min + 1)/256.)
        return image.astype(np.uint8)

    def _look_up_table(self, image, min, max):

        lut = np.arange(2**16, dtype = 'uint16')  # lut = look up table
        lut = self._clip_and_rescale(lut, min, max)

        return np.take(lut, image.astype(np.uint16))  # it s equivalent to lut[image] that is "fancy indexing"



class MahfouzHipHop_metric():

    """Custom metric inspired by the work from Mahfouz et al.

       "A robust method for registration of three-dimensional knee implant models 
       to two-dimensional fluoroscopy images", Mahfouz et al. 2003, IEEE Transactions on Medical Imaging 

       Differently than the Mahfouz_metric, at the update the previously omputed Mahofuz settigns are loaded.

    """


    def __init__(self, metric_info, 
                       PixelType = itk.F,
                       Dimension = 2,
                       ScalarType = itk.D):

        self.info = metric_info
        self.FixedImg = []
        self.MaskImg = []

        # Prepare Gaussian filter for edge image
        self.EdgeGaussSigma = metric_info['edgeGsigma']
        self.EdgeGaussSize = metric_info['edgeGsize']

        # Prepare Canny edge detector for both Moving and Fixed image
        self.MovCannyThr1 = metric_info['MovCannyThr1']
        self.MovCannyThr2 = metric_info['MovCannyThr2']

        # Mahfouz weights
        self.intWeight = metric_info['intWeight']
        self.edgeWeight = metric_info['edgeWeight']

        # Use Mask
        self.useMask = metric_info['useMask']

        # Save images in the folder of the software, with counter starting from 0
        self.saveFixedImages = metric_info['saveFixedImages']
        self.saveMovImages = metric_info['saveMovImages']
        self.temp_counter_fix = 0
        self.temp_counter_mov = 0


    def update(self, itkFixedImage, maskImage, current_settings = None):

        """Sets new fixed image and new mask image"""

        # Load Mahfouz current settings
        FixCannyThr1 = current_settings['CannyThresh1']
        FixCannyThr2 = current_settings['CannyThresh2']
        FixCannyAperture = 2*current_settings['Aperture'] + 1
        minImgContrast = current_settings['minImgContrast']
        maxImgContrast = current_settings['maxImgContrast']

        # Since the HipHop pipeline deals with itk images, itkFixedImage and maskImage
        # need to be converted to numpy arrays for the Mahfouz metric.
        
        # Get COPY OF numpy array of fixed image (not just GetArrayViewFromImage!)
        self.FixedImg = itk.GetArrayFromImage(itkFixedImage)

        # Get Mask
        self.MaskImg = itk.GetArrayFromImage(maskImage)

        # rescale 16bit into 8bit with contrast stretching
        img_for_canny = self._look_up_table(self.FixedImg, minImgContrast, maxImgContrast)

        # generate canny edge fixed image
        FixedEdgeImg = cv2.Canny(img_for_canny, FixCannyThr1, FixCannyThr2,L2gradient=False,apertureSize= FixCannyAperture)  

        # Apply mask to edge image (eliminates edges outside of ROI)
        if self.useMask:
            FixedEdgeImg[self.MaskImg == 0.] = 0.

        # blurr edge image (rescaling to uint16 is necessary)
        self.FixedEdgeImg = cv2.GaussianBlur( FixedEdgeImg*((np.power(2,16) - 1)/(np.power(2,8) - 1)), self.EdgeGaussSize, self.EdgeGaussSigma)

        # Apply mask to Fixed image
        if self.useMask:
            self.FixedImg[self.MaskImg == 0.] = 2.**16 -1.

        if self.saveFixedImages:

            # Save fixed edge images
            cv2.imwrite('FixedEdgImg' + str(self.temp_counter_fix) + '.tif',self.FixedEdgeImg)
            cv2.imwrite('FixedImg' + str(self.temp_counter_fix) + '.tif',self._look_up_table(self.FixedImg, 0, 2**16-1))
            self.temp_counter_fix = self.temp_counter_fix + 1


    def compute(self, movImage):

        """Returns the similarity measure between (masked) fixed image and current (masked) moving image."""

        # Get COPY OF numpy array of moving image (not just GetArrayViewFromImage!)
        movImg = itk.GetArrayFromImage(movImage)

        # generate canny edge moving image
        movEdgeImg = cv2.Canny(movImg.astype(np.uint8), self.MovCannyThr1, self.MovCannyThr2).astype(np.float64) 

        # blurr edge image (rescaling to uint16 is necessary)
        movEdgeImg = cv2.GaussianBlur( movEdgeImg*((np.power(2.,16.) - 1.)/(np.power(2.,8.) - 1.)) , self.EdgeGaussSize, self.EdgeGaussSigma)

        if self.saveMovImages:

            # Save fixed edge images
            cv2.imwrite('MovEdgImg' + str(self.temp_counter_mov) + '.tif',movEdgeImg)
            cv2.imwrite('MovImg' + str(self.temp_counter_mov) + '.tif',movImg.astype(np.uint8))
            self.temp_counter_mov = self.temp_counter_mov + 1
        
        # Compute intensity fitness
        int_fitness = np.sum(np.multiply(self.FixedImg.astype(float),movImg.astype(float)))/np.sum(movImg.astype(float))

        # Compute edge fitness
        edge_fitness = np.sum(np.multiply(self.FixedEdgeImg.astype(float),movEdgeImg.astype(float)))/np.sum(movEdgeImg.astype(float))

        ## Save mov edge image
        #cv2.imwrite('MovEdgImg' + str(self.temp_counter) + '.png',movEdgeImg)
        #self.temp_counter = self.temp_counter + 1

        return -self.intWeight*int_fitness - self.edgeWeight*edge_fitness


    # LOOK UP TABLE: FASTER CONTRAST STRETCHING METHOD (from 16bit to 8bit ONLY)
    def _clip_and_rescale(self, img, min, max):

        image = np.array(img, copy = True) # just create a copy of the array
        image.clip(min,max, out = image)
        image -= min
        image = np.divide(image,(max - min + 1)/256.)
        return image.astype(np.uint16)

    def _look_up_table(self, image, min, max):

        lut = np.arange(2**16, dtype = np.uint16)  # lut = look up table
        lut = self._clip_and_rescale(lut, min, max)

        return np.take(lut, image.astype(np.uint16)).astype(np.uint8)  # it s equivalent to lut[image] that is "fancy indexing"




class MattesMutualv4_metric():

    """Mutual Information based on Mattes et al. (ITK) 
          
        Template for the metric_info:

        {'Name': 'MattesMutualv4', 
            'NumHistBins': 2**4,
            'NormMovImg': False,
            'NormFixImg': False}

        Notes: itk rescales the images so that each drawn pixel contributes to a valid bin,
                hence normalization of the images is not really needed.
    """

    def __init__(self, metric_info, 
                       PixelType = itk.F,
                       Dimension = 2,
                       ScalarType = itk.D):

        self.info = metric_info
        self.PixelType = PixelType
        self.Dimension = Dimension
        self.ScalarType = ScalarType

        # Initialize types
        self.ImageType = itk.Image[self.PixelType, self.Dimension]

        self.MattesMutualv4MetricType = itk.MattesMutualInformationImageToImageMetricv4[self.ImageType,self.ImageType]

        if (self.info['NormFixImg'] == True) or (self.info['NormMovImg'] == True):
            self.NormalizeImageFilterType = itk.NormalizeImageFilter[self.ImageType, self.ImageType]

        # Initialize Identity Transform type
        self.IdentityTransformType = itk.IdentityTransform[self.ScalarType, self.Dimension] 

        # Initialize interpolator for the metric (it does not matter because the transform is the identity)
        self.MetricInterpolatorType = itk.NearestNeighborInterpolateImageFunction[self.ImageType, self.ScalarType]      

        # Initialize Mask Spatial Object
        self.spatialObjectMask = itk.ImageMaskSpatialObject[self.Dimension].New()

        # Create metric
        self.new = self.MattesMutualv4MetricType.New()

        # Set Metric parameters
        #if self.info['AllPixels'] == True: self.new.UseAllPixelsOn()
        if self.info['NumHistBins']: self.new.SetNumberOfHistogramBins(self.info['NumHistBins'])

        # Instantiate transform and interpolator
        identityTransform = self.IdentityTransformType.New()
        interpolatorMetric = self.MetricInterpolatorType.New()

        # Set metric
        self.new.SetTransform(identityTransform)
        self.new.SetMovingInterpolator(interpolatorMetric)

        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter = self.NormalizeImageFilterType.New()
        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter = self.NormalizeImageFilterType.New()


    def update(self, itkFixedImage, maskImage, current_settings = None):

        """Sets new fixed image and new mask image"""

        # Update input fixed image
        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter.SetInput(itkFixedImage)
            self.new.SetFixedImage(self.normalizeFixedImageFilter.GetOutput())
        else: self.new.SetFixedImage( itkFixedImage )

        self.new.SetVirtualDomainFromImage( itkFixedImage ) # SEEMS TO BE OPTIONAL sets parameters of virtual domain from given image (I believe, in case of the Cast Interpolator, it sets the image plane)

        # Update mask
        self.spatialObjectMask.SetImage(maskImage)
        self.new.SetFixedImageMask( self.spatialObjectMask )
        

    def compute(self, movImage):

        """Returns the similarity measure between (masked) fixed image and current (masked) moving image.

           Note: If 'NormMovImg' = True, the given itkFixedImage should be already normalized to zero mean and variance one.
        """

        # Set moving image
        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter.SetInput(movImage)       
            self.new.SetMovingImage(self.normalizeMovingImageFilter.GetOutput())
        else: self.new.SetMovingImage( movImage )
        
        # Initialize (itk)
        self.new.Initialize()
 
        return self.new.GetValue() 


class MattesMutualv3_metric():

    """Mutual information based on Mattes et al (ITK)
          
        Template for the metric_info:

        {'Name': 'MattesMutualv3', 
            'NumHistBins': 2**4, 
            'NumSamples': 1000**2,
            'AllPixels': False,
            'NormMovImg': False,
            'NormFixImg': False}

        Notes: itk rescales the images so that each drawn pixel contributes to a valid bin,
                hence normalization of the images is not really needed.
    """

    def __init__(self, metric_info,
                       PixelType = itk.F,
                       Dimension = 2,
                       ScalarType = itk.D):


        self.info = metric_info
        self.PixelType = PixelType
        self.Dimension = Dimension
        self.ScalarType = ScalarType

        # Initialize types
        self.ImageType = itk.Image[self.PixelType, self.Dimension]

        self.MattesMutualv3MetricType = itk.MattesMutualInformationImageToImageMetric[self.ImageType,self.ImageType]

        if (self.info['NormFixImg'] == True) or (self.info['NormMovImg'] == True):
            self.NormalizeImageFilterType = itk.NormalizeImageFilter[self.ImageType, self.ImageType]

        # Initialize Identity Transform type
        self.IdentityTransformType = itk.IdentityTransform[self.ScalarType, self.Dimension] 

        # Initialize interpolator for the metric (it does not matter because the transform is the identity)
        self.MetricInterpolatorType = itk.NearestNeighborInterpolateImageFunction[self.ImageType, self.ScalarType]      

        # Initialize Mask Spatial Object
        self.spatialObjectMask = itk.ImageMaskSpatialObject[self.Dimension].New()

        # Create metric
        self.new = self.MattesMutualv3MetricType.New()

        # Set Metric parameters
        if self.info['AllPixels'] == True: self.new.UseAllPixelsOn()
        if self.info['NumHistBins']: self.new.SetNumberOfHistogramBins(self.info['NumHistBins'])
        if self.info['NumSamples']: self.new.SetNumberOfSpatialSamples(self.info['NumSamples'])

        # Instantiate transform and interpolator
        self.identityTransform = self.IdentityTransformType.New()
        interpolatorMetric = self.MetricInterpolatorType.New()

        # Set metric
        self.new.SetTransform(self.identityTransform)
        self.new.SetInterpolator(interpolatorMetric)

        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter = self.NormalizeImageFilterType.New()
        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter = self.NormalizeImageFilterType.New()    


    def update(self, itkFixedImage, maskImage, current_settings = None):

        """Sets new fixed image and new mask image"""

        # Update input fixed image
        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter.SetInput(itkFixedImage)
            self.new.SetFixedImage(self.normalizeFixedImageFilter.GetOutput())
        else: self.new.SetFixedImage( itkFixedImage )

        self.new.SetFixedImageRegion( itkFixedImage.GetBufferedRegion()  )

        # Update mask
        self.spatialObjectMask.SetImage(maskImage)
        self.new.SetFixedImageMask( self.spatialObjectMask )


    def compute(self, movImage):

        """Returns the similarity measure between (masked) fixed image and current (masked) moving image.

            If 'NormMovImg' = True, the given itkFixedImage should be already normalized to zero mean and variance one.
        """

        # Set moving image
        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter.SetInput(movImage)       
            self.new.SetMovingImage(self.normalizeMovingImageFilter.GetOutput())
        else: self.new.SetMovingImage( movImage )     

        # Initialize (itk)
        self.new.Initialize()

        p = self.identityTransform.GetParameters()

        return self.new.GetValue(p)


class ViolaMutual_metric():

    """ Mutual information based on Viola et al. (ITK)          
        https://itk.org/ITKExamples/src/Registration/Common/PerformMultiModalityRegistrationWithMutualInformation/Documentation.html

        Template for the metric_info:

        {'Name': 'ViolaMutual', 
            'NumSamples': [], 
            'AllPixels': False, 
            'MovImgStd': 0.4, 
            'FixImgStd': 0.4, 
            'KernelFunction': [],
            'NormFixImg': True}

        The given itkFixedImage should be already normalized to zero mean and variance one.
    """

    def __init__(self, metric_info,
                       PixelType = itk.F,
                       Dimension = 2,
                       ScalarType = itk.D):

        self.info = metric_info
        self.PixelType = PixelType
        self.Dimension = Dimension
        self.ScalarType = ScalarType

        # Initialize types
        self.ImageType = itk.Image[self.PixelType, self.Dimension]
            
        self.ViolaMutualMetricType = itk.MutualInformationImageToImageMetric[self.ImageType,self.ImageType]

        self.NormalizeImageFilterType = itk.NormalizeImageFilter[self.ImageType, self.ImageType]

        # Initialize Identity Transform type
        self.IdentityTransformType = itk.IdentityTransform[self.ScalarType, self.Dimension] 

        # Initialize interpolator for the metric (it does not matter because the transform is the identity)
        self.MetricInterpolatorType = itk.NearestNeighborInterpolateImageFunction[self.ImageType, self.ScalarType]      

        # Initialize Mask Spatial Object
        self.spatialObjectMask = itk.ImageMaskSpatialObject[self.Dimension].New()

        # Create metric
        self.new = self.ViolaMutualMetricType.New() 

        # Metric parameters
        if self.info['AllPixels'] == True: self.new.UseAllPixelsOn()
        if self.info['MovImgStd']: self.new.SetMovingImageStandardDeviation(self.info['MovImgStd'])
        if self.info['FixImgStd']: self.new.SetFixedImageStandardDeviation(self.info['FixImgStd'])
        if self.info['KernelFunction']: self.new.SetKernelFunction(self.info['KernelFunction'])

        # Instantiate transform and interpolator
        self.identityTransform = self.IdentityTransformType.New()
        interpolatorMetric = self.MetricInterpolatorType.New()

        # Set metric
        self.new.SetTransform(self.identityTransform)
        self.new.SetInterpolator(interpolatorMetric)

        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter = self.NormalizeImageFilterType.New()
        # Initialize normalizeMovingImageFilter
        self.normalizeMovingImageFilter = self.NormalizeImageFilterType.New()


    def update(self, itkFixedImage, maskImage, current_settings = None):

        """Sets new fixed image and new mask image"""

        # Update input fixed image
        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter.SetInput(itkFixedImage)
            self.new.SetFixedImage(self.normalizeFixedImageFilter.GetOutput())

            self.normalizeFixedImageFilter.Update()
            fixedImageRegion = self.normalizeFixedImageFilter.GetOutput().GetBufferedRegion() 
            self.new.SetFixedImageRegion( fixedImageRegion )

        else: 
            self.new.SetFixedImage( itkFixedImage )
            self.new.SetFixedImageRegion( itkFixedImage.GetBufferedRegion()  )

        
        if self.info['NumSamples']:
            if self.info['NormFixImg'] == True:
                numSamples = fixedImageRegion.GetNumberOfPixels()*self.info['NumSamples']
            else:
                numSamples = itkFixedImage.GetBufferedRegion().GetNumberOfPixels()*self.info['NumSamples']

            self.new.SetNumberOfSpatialSamples(int(numSamples))             


        # Update mask
        self.spatialObjectMask.SetImage(maskImage)
        self.new.SetFixedImageMask( self.spatialObjectMask )


    def compute(self, movImage):

        """Returns the similarity measure between (masked) fixed image and current (masked) moving image.
 
            The given itkFixedImage should be already normalized to zero mean and variance one.
        """

        # Normalize DRR to zero mean and variance one
        self.normalizeMovingImageFilter.SetInput(movImage)

        # Set moving image
        self.new.SetMovingImage( self.normalizeMovingImageFilter.GetOutput() )

        # Initialize (itk)
        self.new.Initialize()

        p = self.identityTransform.GetParameters()

        return self.new.GetValue(p)



class MeanSquares_metric():

    """Mean squared differences metric (ITK)

       Since multi-modal registration is performed, in this implementation both fixed and moving images are normalized to zero mean and variance one.            
        
    """

    def __init__(self, metric_info,
                       PixelType = itk.F,
                       Dimension = 2,
                       ScalarType = itk.D):

        self.info = metric_info
        self.PixelType = PixelType
        self.Dimension = Dimension
        self.ScalarType = ScalarType

        # Initialize types
        self.ImageType = itk.Image[self.PixelType, self.Dimension]

        self.MeanSquaresMetricType = itk.MeanSquaresImageToImageMetricv4[self.ImageType,self.ImageType]

        self.NormalizeImageFilterType = itk.NormalizeImageFilter[self.ImageType, self.ImageType]

        # Initialize Identity Transform type
        self.IdentityTransformType = itk.IdentityTransform[self.ScalarType, self.Dimension] 

        # Initialize interpolator for the metric (it does not matter because the transform is the identity)
        self.MetricInterpolatorType = itk.NearestNeighborInterpolateImageFunction[self.ImageType, self.ScalarType]      

        # Initialize Mask Spatial Object
        self.spatialObjectMask = itk.ImageMaskSpatialObject[self.Dimension].New()

        # Create metric
        self.new = self.MeanSquaresMetricType.New()

        # Metric parameters (none)

        # Instantiate transform and interpolator
        identityTransform = self.IdentityTransformType.New()
        interpolatorMetric = self.MetricInterpolatorType.New()

        # Set metric
        self.new.SetTransform(self.identityTransform)
        self.new.SetInterpolator(interpolatorMetric)

        self.normalizeMovingImageFilter = self.NormalizeImageFilterType.New()
        self.normalizeFixedImageFilter = self.NormalizeImageFilterType.New()


    def update(self, itkFixedImage, maskImage, current_settings = None):

        """Sets new fixed image and new mask image"""

        # Update input fixed image
        self.normalizeFixedImageFilter.SetInput(itkFixedImage)
        self.new.SetFixedImage(self.normalizeFixedImageFilter.GetOutput())
        self.new.SetVirtualDomainFromImage( itkFixedImage )                

        # Update mask
        self.spatialObjectMask.SetImage(maskImage)
        self.new.SetFixedImageMask( self.spatialObjectMask )


    def compute(self, movImage):
    
        """Returns the similarity measure between (masked) fixed image and current (masked) moving image."""
           
        # Set moving image
        self.normalizeMovingImageFilter.SetInput(movImage)
        self.new.SetMovingImage( self.normalizeMovingImageFilter.GetOutput() )

        # Initialize (itk)
        self.new.Initialize()

        return self.new.GetValue()


class NormCorr_metric():


    """Normalized correlation mtric (ITK)

        Template for the metric_info:

        {'Name': 'NormCorr', 
            'SubtractMeanOn': False,
            'NormFixImg': False,
            'NormMovImg': False}

    """

    def __init__(self, metric_info,
                       PixelType = itk.F,
                       Dimension = 2,
                       ScalarType = itk.D):

        self.info = metric_info
        self.PixelType = PixelType
        self.Dimension = Dimension
        self.ScalarType = ScalarType

        # Initialize types
        self.ImageType = itk.Image[self.PixelType, self.Dimension]

        self.NormCorrMetricType = itk.NormalizedCorrelationImageToImageMetric[self.ImageType,self.ImageType]

        if (self.info['NormFixImg'] == True) or (self.info['NormMovImg'] == True):
            self.NormalizeImageFilterType = itk.NormalizeImageFilter[self.ImageType, self.ImageType]

        # Initialize Identity Transform type
        self.IdentityTransformType = itk.IdentityTransform[self.ScalarType, self.Dimension] 

        # Initialize interpolator for the metric (it does not matter because the transform is the identity)
        self.MetricInterpolatorType = itk.NearestNeighborInterpolateImageFunction[self.ImageType, self.ScalarType]      

        # Initialize Mask Spatial Object
        self.spatialObjectMask = itk.ImageMaskSpatialObject[self.Dimension].New()

        # Create metric
        self.new = self.NormCorrMetricType.New()

        # Metric parameters
        if self.info['SubtractMeanOn'] == True : self.new.SubtractMeanOn()

        # Instantiate transform and interpolator
        self.identityTransform = self.IdentityTransformType.New()
        interpolatorMetric = self.MetricInterpolatorType.New()

        # Set metric
        self.new.SetTransform(self.identityTransform)
        self.new.SetInterpolator(interpolatorMetric)

        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter = self.NormalizeImageFilterType.New()
        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter = self.NormalizeImageFilterType.New()


    def update(self, itkFixedImage, maskImage, current_settings = None):

        """Sets new fixed image and new mask image"""

        # Update input fixed image
        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter.SetInput(itkFixedImage)
            self.new.SetFixedImage(self.normalizeFixedImageFilter.GetOutput())             
        else: 
            self.new.SetFixedImage( itkFixedImage )
        self.new.SetFixedImageRegion( itkFixedImage.GetBufferedRegion() )

        # Update mask
        self.spatialObjectMask.SetImage(maskImage)
        self.new.SetFixedImageMask( self.spatialObjectMask )


    def compute(self, movImage):
    
        """Returns the similarity measure between (masked) fixed image and current (masked) moving image."""

        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter.SetInput(movImage)
            self.new.SetMovingImage(self.normalizeMovingImageFilter.GetOutput())
        else: self.new.SetMovingImage( movImage )
       

        # Initialize (itk)
        self.new.Initialize()

        p = self.identityTransform.GetParameters()

        return self.new.GetValue(p)


class MeanRec_metric():

    """Mean reciprocal squared differences metric (ITK)

        Template for the metric_info:

        {'Name': 'MeanRec', 
            'Lambda': 0.5,
            'NormFixImg': False,
            'NormMovImg': False}

    """

    def __init__(self, metric_info,
                       PixelType = itk.F,
                       Dimension = 2,
                       ScalarType = itk.D):

        self.info = metric_info
        self.PixelType = PixelType
        self.Dimension = Dimension
        self.ScalarType = ScalarType

        # Initialize types
        self.ImageType = itk.Image[self.PixelType, self.Dimension]

        self.MeanRecMetricType = itk.MeanReciprocalSquareDifferenceImageToImageMetric[self.ImageType,self.ImageType]

        if (self.info['NormFixImg'] == True) or (self.info['NormMovImg'] == True):
            self.NormalizeImageFilterType = itk.NormalizeImageFilter[self.ImageType, self.ImageType]

        # Initialize Identity Transform type
        self.IdentityTransformType = itk.IdentityTransform[self.ScalarType, self.Dimension] 

        # Initialize interpolator for the metric (it does not matter because the transform is the identity)
        self.MetricInterpolatorType = itk.NearestNeighborInterpolateImageFunction[self.ImageType, self.ScalarType]      

        # Initialize Mask Spatial Object
        self.spatialObjectMask = itk.ImageMaskSpatialObject[self.Dimension].New()

        # Create metric
        self.new = self.MeanRecMetricType.New()

        # Metric parameters
        if self.info['Lambda']: self.new.SetLambda(self.info['Lambda'])

        # Instantiate transform and interpolator
        self.identityTransform = self.IdentityTransformType.New()
        interpolatorMetric = self.MetricInterpolatorType.New()

        # Set metric
        self.new.SetTransform(self.identityTransform)
        self.new.SetInterpolator(interpolatorMetric)

        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter = self.NormalizeImageFilterType.New()
        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter = self.NormalizeImageFilterType.New()


    def update(self, itkFixedImage, maskImage, current_settings = None):

        """Sets new fixed image and new mask image"""

        # Update input fixed image
        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter.SetInput(itkFixedImage)
            self.new.SetFixedImage(self.normalizeFixedImageFilter.GetOutput())             
        else: 
            self.new.SetFixedImage( itkFixedImage )
        self.new.SetFixedImageRegion( itkFixedImage.GetBufferedRegion() )

        # Update mask
        self.spatialObjectMask.SetImage(maskImage)
        self.new.SetFixedImageMask( self.spatialObjectMask )


    def compute(self, movImage):
    
        """Returns the similarity measure between (masked) fixed image and current (masked) moving image."""

        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter.SetInput(movImage)
            self.new.SetMovingImage(self.normalizeMovingImageFilter.GetOutput())
        else: self.new.SetMovingImage( movImage )


        # Initialize (itk)
        self.new.Initialize()

        p = self.identityTransform.GetParameters()

        return self.new.GetValue(p)



class GradCorr_metric():

    """Gradient COrrelation metric

        Template for the metric_info:

        {'Name': 'GradCorr', 
            'NormFixImg': True,
            'NormMovImg': True,
            'SubtractMeanOn': False}

    """

    def __init__(self, metric_info,
                       PixelType = itk.F,
                       Dimension = 2,
                       ScalarType = itk.D):

        self.info = metric_info
        self.PixelType = PixelType
        self.Dimension = Dimension
        self.ScalarType = ScalarType

        # Initialize types
        self.ImageType = itk.Image[self.PixelType, self.Dimension]

        self.DerivativeImageFilterType = itk.DerivativeImageFilter[self.ImageType, self.ImageType]

        self.NormCorrMetricType = itk.NormalizedCorrelationImageToImageMetric[self.ImageType,self.ImageType]

        if (self.info['NormFixImg'] == True) or (self.info['NormMovImg'] == True):
            self.NormalizeImageFilterType = itk.NormalizeImageFilter[self.ImageType, self.ImageType]

        # Initialize Identity Transform type
        self.IdentityTransformType = itk.IdentityTransform[self.ScalarType, self.Dimension] 

        # Initialize interpolator for the metric (it does not matter because the transform is the identity)
        self.MetricInterpolatorType = itk.NearestNeighborInterpolateImageFunction[self.ImageType, self.ScalarType]      

        # Initialize Mask Spatial Object
        self.spatialObjectMask = itk.ImageMaskSpatialObject[self.Dimension].New()

        # Instantiate metrics
        self.metric_derX  = self.NormCorrMetricType.New() 
        self.metric_derY  = self.NormCorrMetricType.New() 

        # Compute directional gradients
        # Fixed Image
        self.derivativeXfixedImage = self.DerivativeImageFilterType.New()
        self.derivativeYfixedImage = self.DerivativeImageFilterType.New()
        self.derivativeXfixedImage.SetDirection(0)
        self.derivativeYfixedImage.SetDirection(1)
        # Moving Image
        self.derivativeXmovingImage = self.DerivativeImageFilterType.New()
        self.derivativeYmovingImage = self.DerivativeImageFilterType.New()
        self.derivativeXmovingImage.SetDirection(0)
        self.derivativeYmovingImage.SetDirection(1)

        # Metrics parameters
        if self.info['SubtractMeanOn'] == True : 
            self.metric_derX.SubtractMeanOn()
            self.metric_derY.SubtractMeanOn()

        # Instantiate transform and interpolator
        self.identityTransform = self.IdentityTransformType.New()
        interpolatorMetric = self.MetricInterpolatorType.New()

        self.metric_derX.SetTransform(self.identityTransform)
        self.metric_derX.SetInterpolator(interpolatorMetric)
        self.metric_derY.SetTransform(self.identityTransform)
        self.metric_derY.SetInterpolator(interpolatorMetric)

        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter = self.NormalizeImageFilterType.New()
        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter = self.NormalizeImageFilterType.New()

        # Save images in the folder of the software, with counter starting from 0
        self.saveFixedImages = metric_info['saveFixedImages']
        self.saveMovImages = metric_info['saveMovImages']
        self.temp_counter_fix = 0
        self.temp_counter_mov = 0

    def update(self, itkFixedImage, maskImage, current_settings = None):

        """Sets new fixed image and new mask image"""

        # Update input fixed image
        if self.info['NormFixImg'] == True:
            self.normalizeFixedImageFilter.SetInput(itkFixedImage)
            self.derivativeXfixedImage.SetInput(self.normalizeFixedImageFilter.GetOutput())     
            self.derivativeYfixedImage.SetInput(self.normalizeFixedImageFilter.GetOutput())       
        else: 
            self.derivativeXfixedImage.SetInput(itkFixedImage)
            self.derivativeYfixedImage.SetInput(itkFixedImage)        

        self.metric_derX.SetFixedImage(self.derivativeXfixedImage.GetOutput())
        self.metric_derY.SetFixedImage(self.derivativeYfixedImage.GetOutput())

        self.metric_derX.SetFixedImageRegion( itkFixedImage.GetBufferedRegion() )
        self.metric_derY.SetFixedImageRegion( itkFixedImage.GetBufferedRegion() )


        # Update mask
        self.spatialObjectMask.SetImage(maskImage)
        self.metric_derX.SetFixedImageMask( self.spatialObjectMask )
        self.metric_derY.SetFixedImageMask( self.spatialObjectMask )

        if self.saveFixedImages:

            # Write rescaled fixed image
            WriterTypeOutput = itk.ImageFileWriter[self.ImageType]
            writerMovingImage = WriterTypeOutput.New()
            writerMovingImage.SetInput(self.derivativeXfixedImage.GetOutput())
            writerMovingImage.SetFileName('derivativeXfixedImage_' + str(self.temp_counter_fix) + '.mha')
            writerMovingImage.Update()

            WriterTypeOutput = itk.ImageFileWriter[self.ImageType]
            writerMovingImage = WriterTypeOutput.New()
            writerMovingImage.SetInput(self.derivativeYfixedImage.GetOutput())
            writerMovingImage.SetFileName('derivativeYfixedImage_' + str(self.temp_counter_fix) + '.mha')
            writerMovingImage.Update()

            self.temp_counter_fix = self.temp_counter_fix + 1
             

    def compute(self, movImage):
    
        """Returns the similarity measure between (masked) fixed image and current (masked) moving image."""
        
        if self.info['NormMovImg'] == True:
            self.normalizeMovingImageFilter.SetInput(movImage)
            self.derivativeXmovingImage.SetInput(self.normalizeMovingImageFilter.GetOutput())
            self.derivativeYmovingImage.SetInput(self.normalizeMovingImageFilter.GetOutput())
        else: 
            self.derivativeXmovingImage.SetInput(movImage)
            self.derivativeYmovingImage.SetInput(movImage)

        self.metric_derX.SetMovingImage(self.derivativeXmovingImage.GetOutput())
        self.metric_derY.SetMovingImage(self.derivativeYmovingImage.GetOutput())

        ## Write gradient images
        ## Write rescaled fixed image
        #WriterTypeOutput = itk.ImageFileWriter[self.ImageType]
        #writerMovingImage = WriterTypeOutput.New()
        #writerMovingImage.SetInput(self.derivativeXfixedImage.GetOutput())
        #writerMovingImage.SetFileName('derivativeXfixedImage.mha')
        #writerMovingImage.Update()

        #WriterTypeOutput = itk.ImageFileWriter[self.ImageType]
        #writerMovingImage = WriterTypeOutput.New()
        #writerMovingImage.SetInput(self.derivativeYfixedImage.GetOutput())
        #writerMovingImage.SetFileName('derivativeYfixedImage.mha')
        #writerMovingImage.Update() 
 
        if self.saveMovImages:

            # Write rescaled moving image
            WriterTypeOutput = itk.ImageFileWriter[self.ImageType]
            writerMovingImage = WriterTypeOutput.New()
            writerMovingImage.SetInput(self.derivativeXmovingImage.GetOutput())
            writerMovingImage.SetFileName('derivativeXmovingImage_'+ str(self.temp_counter_mov) + '.mha')
            writerMovingImage.Update()

            WriterTypeOutput = itk.ImageFileWriter[self.ImageType]
            writerMovingImage = WriterTypeOutput.New()
            writerMovingImage.SetInput(self.derivativeYmovingImage.GetOutput())
            writerMovingImage.SetFileName('derivativeYmovingImage_'+ str(self.temp_counter_mov) + '.mha')
            writerMovingImage.Update()      

            self.temp_counter_mov = self.temp_counter_mov + 1
                  

        # Initialize (itk)
        self.metric_derX.Initialize()
        self.metric_derY.Initialize()

        p = self.identityTransform.GetParameters()

        return (self.metric_derX.GetValue(p) + self.metric_derY.GetValue(p))/2.
