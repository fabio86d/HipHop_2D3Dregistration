####  PYTHON MODULES
import itk
import numpy as np
import time
import os
import sys
import glob
import json
import csv
import pandas as pd


#### MY MODULES
import ReadWriteImageModule as rw
import RigidMotionModule as rm
import ProjectorsModule as pm
import MetricsModule as mm
import OptimizersModule as om


class HipHop():

    """Fabio D'Isidoro - ETH Zurich - March 2018

       A framework for 2D/3D registration between a CT or STL 3D model 
       and a 2D X-ray fluoroscopic image.

    """

    def __init__(self, ProjectDir,
                       ADL_info,
                       Projector_info,
                       Metrics_info,                   
                       Optimizer_info,
                       GroundTruthPose = False,
                       pixel_type = itk.F,
                       scalar_type = itk.D,
                       mask_pixel_type = itk.UC,
                       png_pixel_type = itk.UC,
                       input_dimension = 3, 
                       metric_dimension = 2,
                       fixedImgExtension = '.tif'):

        print('-----------------------', '\n Hallo HipHop! \n-----------------------')

        # Get parameters from input arguments
        self.ProjectDir = ProjectDir  
        self.PatientID = ADL_info['PatientID']
        self.ADLType = ADL_info['ActivityType']
        self.TrialName = ADL_info['TrialName']
        self.PixelType = pixel_type
        self.ScalarType = scalar_type
        self.MaskPixelType = mask_pixel_type 
        self.OutputPngPixelType = png_pixel_type
        self.InputDimension = input_dimension
        self.MetricDimension = metric_dimension
        self.fixedImgExt = fixedImgExtension
        self.FixedImageType = itk.Image[pixel_type, metric_dimension]
        self.MaskImageType = itk.Image[mask_pixel_type, metric_dimension]

        # Initialize variables
        self.OutputsDir = [] 
        self.Part = []
        self.FixedImage = []
        self.MaskImage = []
        self.Metrics = []
        self.Projector = []
        self.Optimizer = []
        self.GroundTruthPose = []

        # Define directories and filepaths
        self.CameraParamFilepath = self.ProjectDir + '\\' + self.PatientID + '\\analysis\\calibrations\\' + self.PatientID + '_camera_intrinsic_parameters.txt'
        self.RegistrationDir = self.ProjectDir + '\\' + self.PatientID + '\\analysis\\registration\\HipHop\\'
        self.FixedImagesDir = self.ProjectDir + '\\' + self.PatientID + '\\fixed_images\\'
        self.ProcFixedImagesDir = self.FixedImagesDir + 'HipHopReg\\'
        self.MasksDir = self.ProcFixedImagesDir + 'Masks\\'
        self.ImgParamDir = self.ProcFixedImagesDir + 'ImgParam\\'       
        self.OptHistoryDir = self.RegistrationDir + 'OptInfo\\'
        self.InitPosesDir = self.RegistrationDir + 'Initialization\\'

        # Initialize projector
        self._initialize_projector(Projector_info)

        # Initialize metric
        self._initialize_metrics(Metrics_info)

        # Initialize optimizer
        self._initialize_optimizer(Optimizer_info)

        # Initialize images to be registered
        self._initialize_images()   

        # Get ground truth poses (if available)
        if GroundTruthPose:
            self.GroundTruthDir = self.ProjectDir + '\\' + self.PatientID + '\\analysis\\registration\\HipHop\\Ground_Truth_Poses\\'
    


    def _initialize_metrics(self, Metrics_info):

        # Generate list of similarity metrics
        self._metrics_info = Metrics_info
        for metrics_info in Metrics_info:

            self.Metrics.append( mm.metrics_factory(metrics_info,
                                                    PixelType = self.PixelType,
                                                    Dimension = self.MetricDimension,
                                                    ScalarType = self.ScalarType) )

        # Check that list_of_metric_objects is not empty   
        assert(self.Metrics), sys.exit('No metric has been initialized') 

        print(' Metrics initialized')



    def _initialize_projector(self, Projector_info):

        # Load camera intrinsic parameters
        intrinsic_parameters = np.genfromtxt(self.CameraParamFilepath, delimiter=',', usecols = [1])
        Projector_info['focal_lenght'] = intrinsic_parameters[0]
        Projector_info['DRRspacing_x'] = intrinsic_parameters[1]
        Projector_info['DRRspacing_y'] = intrinsic_parameters[2]
        Projector_info['DRR_ppx'] = intrinsic_parameters[3]
        Projector_info['DRR_ppy'] = intrinsic_parameters[4]
        Projector_info['DRRsize_x'] = int(intrinsic_parameters[5])
        Projector_info['DRRsize_y'] = int(intrinsic_parameters[6])

        # Create Projector attribute of class HipHop
        self.Model3D = Projector_info['3Dmodel']
        model_filepath = glob.glob(self.ProjectDir + '\\' + self.PatientID + '\\3D_models\\ready\\' + self.PatientID + '_' + self.Model3D + '*')[0]
        self.Projector = pm.projector_factory(Projector_info, model_filepath)

        # Initialize types
        self.OutputPngImageType = itk.Image[self.OutputPngPixelType, self.MetricDimension]
        self.Write2DImageType = itk.Image[self.PixelType, self.MetricDimension]
        self.RescaleFilterType = itk.RescaleIntensityImageFilter[self.FixedImageType, self.OutputPngImageType]

        # Initialize DRR counter
        self.DRR_counter = 1

        print(' Projector initialized')



    def _initialize_optimizer(self, Optimizer_info):

        if Optimizer_info['Norm']:
            self.Optimizer = om.optimizer_factory(Optimizer_info, self._cost_function_norm)
        else:
            self.Optimizer = om.optimizer_factory(Optimizer_info, self._cost_function)
        self.OptHistory = {'Img#': [], 'To_register': [], 'Registered': [], 'To_correct': []}

        print('\n Optimizer initialized')



    def _initialize_images(self):

        # Finds all images to be registered
        fixed_images_names = self.FixedImagesDir + self.TrialName + '*' + self.fixedImgExt
        self.fixed_images = sorted(glob.glob(fixed_images_names))
        self.num_fixed_images = len(self.fixed_images)



    def _initialize_GroundTruth(self, GroundTruthFile,
                                      fixed_image_filename,
                                      eulerSequence = 'zxy'):

        """ Initializes ground truth pose for the given image, 
            generates the ground truth DRR and returns the metric(s) for the ground truth pose
        """

        print('\n Computing metric(s) for Ground Truth Pose')

        # Load ground truth pose
        ground_truth_transform = np.loadtxt(GroundTruthFile, delimiter = ';') 

        # Get ground truth for ZXY Euler and for translations
        if eulerSequence == 'zxy':
            Euler_zxy_g, e2 = rm.get_euler_zxy(ground_truth_transform[:3, :3]) 
            rotx_gt = np.deg2rad(Euler_zxy_g[1])
            roty_gt = np.deg2rad(Euler_zxy_g[2])
            rotz_gt = np.deg2rad(Euler_zxy_g[0])

        t = ground_truth_transform[:3, 3]
        tx_gt = t[0]
        ty_gt = t[1]
        tz_gt = t[2]         

        # Create dict for ground truth pose
        self.GroundTruthPose = {'EulerSequence': eulerSequence, 'Rotx': rotx_gt, 'Roty': roty_gt, 'Rotz': rotz_gt, 
                            'Translx': tx_gt, 'Transly': ty_gt, 'Translz': tz_gt}

        # Compute DRR and metric for ground truth pose
        transform_parameters_gt = np.array([rotx_gt, roty_gt, rotz_gt, tx_gt, ty_gt, tz_gt])
        metric_values = self._compute_metric(transform_parameters_gt, save_DRR = True, save_id = fixed_image_filename + '_GroundTruthPose_' + self.Model3D)

        # Save metric for ground truth
        text_file = open(self.CurrentOutputsDir + '\\Metrics\\' + fixed_image_filename + '_GroundTruthPose_' + self.Model3D + '.txt', "w")
        text_file.write('Fixed Image: ' + fixed_image_filename + '\n')
        text_file.write('Ground Truth parameters: ' + str(transform_parameters_gt) + '\n')
        iter_metric = 0
        for metric in self.Metrics:
            text_file.write(str(iter_metric) + ': Name ' + metric.info['Name'] + ', ')
            for key in metric.info.keys():
                if key != 'Name': text_file.write(key + ' ' + str(metric.info[key]) + ',')
            text_file.write(' Metric value Ground Truth Pose: ' + str(metric_values[iter_metric]) + '\n')
            iter_metric = iter_metric + 1
        text_file.close()  

        return transform_parameters_gt, metric_values


    def _initialize_Pose(self, InitPoseFile,
                               fixed_image_filename,
                               eulerSequence = 'zxy'):

        """ Initializes pose for current image and computes relative DRR """

        # Load initialization pose
        init_transform = np.loadtxt(InitPoseFile, delimiter = ';') 

        # Get ground truth for ZXY Euler and for translations
        if eulerSequence == 'zxy':
            Euler_zxy_g, e2 = rm.get_euler_zxy(init_transform[:3, :3]) 
            rotx_init = np.deg2rad(Euler_zxy_g[1])
            roty_init = np.deg2rad(Euler_zxy_g[2])
            rotz_init = np.deg2rad(Euler_zxy_g[0])

        t = init_transform[:3, 3]
        tx_init = t[0]
        ty_init = t[1]
        tz_init = t[2]  

        # Computes DRR and metric for initialized pose
        transform_parameters_init = np.array([rotx_init, roty_init, rotz_init, tx_init, ty_init, tz_init])
        self._generateDRR(transform_parameters_init, save_DRR = True, save_id = fixed_image_filename + '_InitializationPose_' + self.Model3D)                

        return transform_parameters_init



    def _delete_me(self):

        """ To be called  every time the object HipHop has to be deleted """

        # Delete projector and free GPU!
        self.Projector.delete()



    def _generateDRR(self, transform_parameters, 
                           save_DRR = False,                         
                           save_id = []):

        """ Generates a DRR for the given parameters (used by default as ZXY a Euler sequence) and saves it if needed. """

        # Compute DRR
        DRR = self.Projector.compute(transform_parameters)


        # Save DRR
        if save_DRR and save_id:  
 
            #rw.ImageWriter( DRR, self.Write2DImageType, self.CurrentOutputsDir + '\\DRRs\\' + str(save_id), extension = '.mha')   

            # Rescale write as 8 bit .png
            rescaler = self.RescaleFilterType.New()
            rescaler.SetOutputMinimum(0)
            rescaler.SetOutputMaximum(255)
            rescaler.SetInput(DRR)
            rw.ImageWriter(rescaler.GetOutput(), self.OutputPngImageType, self.CurrentOutputsDir + '\\DRRs\\' + str(save_id), extension = '.tif')   

        return DRR



    def _update_metric(self, fixed_image_filename):

        """ Updates the metrics attribute with the current image and relative mask. """ 

        FixImgFilepath = glob.glob(self.ProcFixedImagesDir + fixed_image_filename + '*.mha')[0]
        MaskImgFilepath = glob.glob(self.MasksDir + fixed_image_filename + '_' + self.Model3D + '*.mha')[0]

        # Read fixed image
        if FixImgFilepath:
            FixedImage = rw.ImageReader(FixImgFilepath, self.FixedImageType, compute_info = False)
        else:
            sys.exit('Processed Fixed Image not found!', FixImgFilepath)

        # Read Mask Image
        if MaskImgFilepath:
            MaskImage = rw.ImageReader(MaskImgFilepath, self.MaskImageType, compute_info = False)
        else:
            sys.exit('Processed Fixed Image not found!')

        # Load current settings (only for MahfouzHipHop similarity measure)
        if self.Metrics[0].info['Name'] == 'MahfouzHipHop':
            current_settings = np.load(self.ImgParamDir + 'ImgParam_' + fixed_image_filename + '_' + self.Part + '.npy').item()
        else:
            current_settings = None

        # Update metrics with current Image and relative Mask (and image-specific settings, if needed)
        for metric in self.Metrics:
            metric.update(FixedImage, MaskImage, current_settings = current_settings) 


    def _compute_metric(self, transform_parameters, 
                              save_DRR = False,
                              save_id = []):

        """ Computes the metric(s) between the current Image and the DRR
            generated for the specified set of pose parameter. 
            Returns a vector of metric values as long as the the number of Metrics.
        """   

        # Generate DRR
        DRR = self._generateDRR(transform_parameters, save_DRR, save_id)

        # Compute metric(s)       
        metric_values = [[] for i in range(len(self.Metrics))]
        iter_metric = 0
        for metric in self.Metrics:

            metric_values[iter_metric] = metric.compute(DRR)
            iter_metric = iter_metric + 1
        
        return metric_values



    def _cost_function(self, p, grad = []):

        """ Objective function to be given as input to the optimizer. 
            The parameters vector p has to be provided with the following order: 
            Rotx, Roty, Rotz, Translx, Transly, Translz
            If more metrics are provided, the first one is chosen by default.
        """
        
        # Update transform parameters
        self._opt_parameters[self.Optimizer.to_optimize] = p

        # Generate DRR        
        DRR = self._generateDRR(self._opt_parameters, save_DRR = False, save_id = self.DRR_counter)
        
        # Compute metric (first one only)
        metric_value = self.Metrics[0].compute(DRR)

        self.DRR_counter += 1

        return metric_value



    def _cost_function_norm(self, p, grad = []):

        """ Objective function to be given as input to the optimizer. 
            The Normalized parameters vector p has to be provided with the following order: 
            Rotx, Roty, Rotz, Translx, Transly, Translz
            If more metrics are provided, the first one is chosen by default.
        """

        # Update transform parameters with de-normalized values
        p = list(np.multiply((np.asarray(self._uplim) - np.asarray(self._lowlim)),p) + np.asarray(self._lowlim))
        self._opt_parameters[self.Optimizer.to_optimize] = p

        # Generate DRR
        DRR = self._generateDRR(self._opt_parameters, save_DRR = False, save_id = self.DRR_counter)

        # Compute metric (first one only)
        metric_value = self.Metrics[0].compute(DRR) 
    
        return metric_value



    def _run_case_study(self, input_transform_parameters,
                              FixedImageSubDir, 
                              GroundTruthPose,
                              save_DRRs):

        """
            Computes (and saves) the DRR(s) for the specified list of transform parameters 
            (as offset from Ground Truth transformation if specified so).

            input_transform_parameters must be a list, even if with only one set of parameters.

        """  

        print('-----------------------', '\n Case Study')

        # Loads all fixed images in the specified sub-directory for fixed images.
        fixed_image_path = self.ProjectDir + '\\fixed_images\\' + FixedImageSubDir

        for fixed_image_file in os.listdir(fixed_image_path):
            if fixed_image_file.endswith(self.fixedImgExt):  

                fixed_image_filename = os.path.splitext(fixed_image_file)[0]
                print('\n Fixed Image: ', fixed_image_filename, '\n')
               
                # Update metrics with current fixed image and mask image
                self._update_metric(fixed_image_path, fixed_image_filename) 

                # Set current Ground truth pose
                if GroundTruthPose:
                    GroundTruthFile = fixed_image_path + '\\ground_truth_poses\\ready\\gt_' + fixed_image_filename + '.csv'
                    gt = self._initialize_GroundTruth(GroundTruthFile, fixed_image_filename)

                # Go through list of transform parameters
                print('  Computing metrics for input transforms')
                save_id = 0
                metric_values = [[] for i in range(np.shape(input_transform_parameters)[0])]
                for i in range(np.shape(input_transform_parameters)[0]):
                    transform_parameters = input_transform_parameters[i]
                    
                    if GroundTruthPose:
                        transform_parameters = np.add(transform_parameters, gt)

                    #print('  Param ', transform_parameters)
                    # Compute metric(s)
                    metric_values[i] = self._compute_metric(transform_parameters, save_DRR = save_DRRs, save_id = fixed_image_filename + '_case' + str(save_id))

                    save_id = save_id + 1                

                # Save metrics
                text_file = open(self.OutputsDir + '\\Metrics\\' + fixed_image_filename + '_CaseStudy.txt', "w") 
                text_file.write('Fixed Image: ' + fixed_image_filename + '\n')
                for i in range(len(input_transform_parameters)):
                    if GroundTruthPose: text_file.write('\nID:' + str(i) + ', Parameters (offset from Ground Truth): ' + str(input_transform_parameters[i]) + '\n')       
                    else: text_file.write('\nID:' + str(i) + ', Parameters: ' + str(input_transform_parameters[i]) + '\n') 
                    iter_metric = 0
                    for metric in self.Metrics:
                        text_file.write(' ' + str(iter_metric) + ': Name ' + metric.info['Name'] + ', ')
                        for key in metric.info.keys():
                            if key != 'Name': text_file.write(key + ' ' + str(metric.info[key]) + ',')
                        text_file.write(' Metric value: ' + str(metric_values[i][iter_metric]) + '\n')
                        iter_metric = iter_metric + 1
                text_file.close() 
                

    def _initialize_landscape(self, landscape_dim, 
                                    domain_parameters,
                                    GroundTruthPose):

        """
            Given a dict of domain_parameters, computes the range of pose parameters for the compuations of the 
            1 dim Landscape or of the 6 dim Landscape (by default around the ground truth pose)
 
        """  

        # Assign centers as ground truth pose (if so)
        if GroundTruthPose:
                       
            domain_parameters['Translx']['Centre'] = self.GroundTruthPose['Translx']
            domain_parameters['Transly']['Centre'] = self.GroundTruthPose['Transly']
            domain_parameters['Translz']['Centre'] = self.GroundTruthPose['Translz']
            domain_parameters['Rotx']['Centre'] = self.GroundTruthPose['Rotx']
            domain_parameters['Roty']['Centre'] = self.GroundTruthPose['Roty']
            domain_parameters['Rotz']['Centre'] = self.GroundTruthPose['Rotz']
       
        # Generate ranges linspace(-width, + width, NSamples+1) making sure that the centre is included
        if domain_parameters['Translx']['NSamples'] == 2:
            self.range_tx = np.array([domain_parameters['Translx']['Centre'], domain_parameters['Translx']['Centre'] + domain_parameters['Translx']['Width']])
        else:
            a1_tx = np.linspace(domain_parameters['Translx']['Centre'] - domain_parameters['Translx']['Width'],domain_parameters['Translx']['Centre'],int(round(domain_parameters['Translx']['NSamples']/2.)), endpoint = False)
            a2_tx = np.linspace(domain_parameters['Translx']['Centre'], domain_parameters['Translx']['Centre'] + domain_parameters['Translx']['Width'], int(round(domain_parameters['Translx']['NSamples']/2.)+1))
            self.range_tx = np.concatenate([a1_tx, a2_tx])

        if domain_parameters['Transly']['NSamples'] == 2:
            self.range_ty = np.array([domain_parameters['Transly']['Centre'], domain_parameters['Transly']['Centre'] + domain_parameters['Transly']['Width']])
        else:
            a1_ty = np.linspace(domain_parameters['Transly']['Centre'] - domain_parameters['Transly']['Width'],domain_parameters['Transly']['Centre'],int(round(domain_parameters['Transly']['NSamples']/2.)), endpoint = False)
            a2_ty = np.linspace(domain_parameters['Transly']['Centre'], domain_parameters['Transly']['Centre'] + domain_parameters['Transly']['Width'], int(round(domain_parameters['Transly']['NSamples']/2.)+1))
            self.range_ty = np.concatenate([a1_ty, a2_ty])

        if domain_parameters['Translz']['NSamples'] == 2:
            self.range_tz = np.array([domain_parameters['Translz']['Centre'], domain_parameters['Translz']['Centre'] + domain_parameters['Translz']['Width']])
        else:
            a1_tz = np.linspace(domain_parameters['Translz']['Centre'] - domain_parameters['Translz']['Width'],domain_parameters['Translz']['Centre'],int(round(domain_parameters['Translz']['NSamples']/2.)), endpoint = False)
            a2_tz = np.linspace(domain_parameters['Translz']['Centre'], domain_parameters['Translz']['Centre'] + domain_parameters['Translz']['Width'], int(round(domain_parameters['Translz']['NSamples']/2.)+1))
            self.range_tz = np.concatenate([a1_tz, a2_tz])

        if domain_parameters['Rotx']['NSamples'] == 2:
            self.range_rotx = np.array([domain_parameters['Rotx']['Centre'], domain_parameters['Rotx']['Centre'] + domain_parameters['Rotx']['Width']])
        else:
            a1_rx = np.linspace(domain_parameters['Rotx']['Centre'] - domain_parameters['Rotx']['Width'],domain_parameters['Rotx']['Centre'],int(round(domain_parameters['Rotx']['NSamples']/2.)), endpoint = False)
            a2_rx = np.linspace(domain_parameters['Rotx']['Centre'], domain_parameters['Rotx']['Centre'] + domain_parameters['Rotx']['Width'], int(round(domain_parameters['Rotx']['NSamples']/2.)+1))
            self.range_rotx = np.concatenate([a1_rx, a2_rx])

        if domain_parameters['Roty']['NSamples'] == 2:
            self.range_roty = np.array([domain_parameters['Roty']['Centre'], domain_parameters['Roty']['Centre'] + domain_parameters['Roty']['Width']])
        else:
            a1_ry = np.linspace(domain_parameters['Roty']['Centre'] - domain_parameters['Roty']['Width'],domain_parameters['Roty']['Centre'] ,int(round(domain_parameters['Roty']['NSamples']/2.)), endpoint = False)
            a2_ry = np.linspace(domain_parameters['Roty']['Centre'] , domain_parameters['Roty']['Centre'] + domain_parameters['Roty']['Width'], int(round(domain_parameters['Roty']['NSamples']/2.)+1))
            self.range_roty = np.concatenate([a1_ry, a2_ry])

        if domain_parameters['Rotz']['NSamples'] == 2:
            self.range_rotz = np.array([domain_parameters['Rotz']['Centre'], domain_parameters['Rotz']['Centre'] + domain_parameters['Rotz']['Width']])
        else:
            a1_rz = np.linspace(domain_parameters['Rotz']['Centre'] -domain_parameters['Rotz']['Width'],domain_parameters['Rotz']['Centre'],int(round(domain_parameters['Rotz']['NSamples']/2.)), endpoint = False)
            a2_rz = np.linspace(domain_parameters['Rotz']['Centre'], domain_parameters['Rotz']['Centre'] + domain_parameters['Rotz']['Width'], int(round(domain_parameters['Rotz']['NSamples']/2.)+1))
            self.range_rotz = np.concatenate([a1_rz, a2_rz])

        # generates centres
        self.centers = np.array([domain_parameters['Rotx']['Centre'], 
                                 domain_parameters['Roty']['Centre'], 
                                 domain_parameters['Rotz']['Centre'], 
                                 domain_parameters['Translx']['Centre'], 
                                 domain_parameters['Transly']['Centre'], 
                                 domain_parameters['Translz']['Centre']])

        # Initialize cost function
        if landscape_dim == 1:

            # Initialize 1dim cost function (coluns [2:end] are metric values for the metric in metric_info_list
            self.cost_function_1dim = {'Translx': np.zeros((len(self.range_tx), len(self.Metrics) + 1)),
                                       'Transly': np.zeros((len(self.range_ty), len(self.Metrics) + 1)),
                                       'Translz': np.zeros((len(self.range_tz), len(self.Metrics) + 1)),
                                       'Rotx': np.zeros((len(self.range_rotx), len(self.Metrics) + 1)),
                                       'Roty': np.zeros((len(self.range_roty), len(self.Metrics) + 1)),
                                       'Rotz': np.zeros((len(self.range_rotz), len(self.Metrics) + 1)) }
           
            # Saves metric list info
            self._save_info_1dim_landscape(self._metrics_info, domain_parameters)
            
            
            return [self.range_rotx, self.range_roty, self.range_rotz, self.range_tx, self.range_ty, self.range_tz]


        elif landscape_dim == 6:

            # Check that only one metric is given
            assert (len(self._metrics_info) == 1), sys.exit('6dim Landscape can be computed for one metric only!')

            # Initialize 6dim cost function for the only single metric
            nvalues = len(self.range_rotx)*len(self.range_roty)*len(self.range_rotz)*len(self.range_tx)*len(self.range_ty)*len(self.range_tz)
            self.cost_function_6dim = np.zeros((nvalues, (len(domain_parameters) + 1)))   

            # Saves metric list info
            self._save_info_6dim_landscape(domain_parameters)



    def _save_info_1dim_landscape(self, metrics_info, domain_parameters):

        """
            Saves:
            - domain parameters and metrics info (for plots)
            - info txt that explains which metric (with which which metric parameters) was used for each column of the 1 dim cost function (apart from the last one).

        """
        
        # Save domain parameters (for plots)
        np.save(self.CurrentOutputsDir + '\\domain_parameters_1dim.npy', domain_parameters)
 
        # Save info metric txt
        text_file = open(self.CurrentOutputsDir + '\\Metrics\\Metrics_list.txt', "w")
        text_file.write('Metrics description \n')
        iter_metric = 1
        for metric in self.Metrics:
            text_file.write(str(iter_metric) + ': Name ' + metric.info['Name'] + ', ')
            for key in metric.info.keys():
                if key != 'Name': text_file.write(key + ' ' + str(metric.info[key]) + ',')
            iter_metric = iter_metric + 1
            text_file.write('\n')
        text_file.close()  

        # Save info metric npy (for plots)
        with open(self.CurrentOutputsDir + '\\Metrics\\Metrics_list.npy', 'w') as fout:
            json.dump(metrics_info, fout)


    def _save_info_6dim_landscape(self, domain_parameters):  

        """
            Saves domain parameters for the 6dim landscape

        """  

        metric = self._metrics_info[0]

        # Save domain parameters (for plots)
        np.save(self.CurrentOutputsDir + '\\domain_parameters_6dim.npy', domain_parameters)
 
        # Save info metric
        text_file = open(self.CurrentOutputsDir + '\\Metrics\\Metric_info_' + metric['Name'] + '.txt', "w")
        text_file.write('Metric description: \n')
        text_file.write(metric['Name'] + ', ')
        for key in metric.keys():
            if key != 'Name': text_file.write(key + ' ' + str(metric[key]) + ',')
        text_file.close() 

        # Save info metric npy (for plots)
        with open(self.CurrentOutputsDir + '\\Metrics\\Metric_info_' + metric['Name']  + '.npy', 'w') as fout:
            json.dump(metric, fout)  


    def _save_1dim_landscape(self, which_parameter):

        """
            Saves the computed 1dim landscape for a single parameter, as a csv file

        """ 
        
        # Collect metric names and assign index
        hdr = which_parameter
        i = 0
        for metric in self.Metrics:            
            hdr = hdr + '_' + str(i) + '_' + metric.info['Name']
            i = i + 1

        # Save cost function
        np.savetxt(self.CurrentOutputsDir + '\\Metrics\\oneDimLandscape_' + which_parameter + '.csv', self.cost_function_1dim[which_parameter] , delimiter = ',', newline='\n', header = hdr)


    def _save_6dim_landscape(self):

        """
            Saves the computed 6dim landscape

        """ 

        # Column of row titles
        hdr = 'RotX,RotY,RotZ,TransX,TransY,TransZ,MetricValue'

        # Save cost function
        metric = self.Metrics[0]
        np.savetxt(self.CurrentOutputsDir + '\\Metrics\\sixDimLandscape_' + metric.info['Name'] + '.csv', self.cost_function_6dim , delimiter = ',', newline='\n', header = hdr)


    def _compute_minimum_6dim_landscape(self):

        """
            Computes the transform parameters relative to the global min of the 6dim landscape
            and generates a DRR out of it.

        """ 

        # Retrieve transform parameters for minimum
        minimum = self.cost_function_6dim[ np.argmin(self.cost_function_6dim[:,6]) ]
        minimum_parameters = minimum[0:6]

        print('\n Global minimum (', minimum[6] ,') found for parameters: ', minimum_parameters)

        # Generates DRR for global minimum
        metric = self.Metrics[0]
        s = self.CurrentOutputsDir + '\\DRRs\\' + metric.info['Name']  

        self._generateDRR(minimum_parameters, save_DRR = True, save_id = metric.info['Name'] + '_minimum')


    def _run_1dim_landscape(self, domain_parameters, 
                                  FixedImages,  
                                  GroundTruthPose, 
                                  save_DRRs):

        """
            Computes 1 dim Landscape relative to a set of similarity metrics, for a set of fixed images and given one range of parameters:
            - domain_parameters = dict {'Translx': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Transly': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Translz': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Rotx': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Roty': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Rotz': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}}

        """  

        landscape_dim = 1

        # Loads specified fixed images
        for fixed_image_name in FixedImages:

            print('-----------------------', '\n Computation of 1dim Landscape for Fixed Image: ', fixed_image_name, '\n')

            # Overwrite OutputsDir with sub-folder relative to current fixed image
            self.CurrentOutputsDir = self.OutputsDir + '\\' + fixed_image_name
            if not os.path.isdir(self.CurrentOutputsDir):
                os.makedirs(self.CurrentOutputsDir)
            if not os.path.isdir(self.CurrentOutputsDir + '\\DRRs'):
                os.makedirs(self.CurrentOutputsDir + '\\DRRs')  # DRRs folder
            if not os.path.isdir(self.CurrentOutputsDir + '\\Metrics'):
                os.makedirs(self.CurrentOutputsDir + '\\Metrics') # Metrics folder

            # Update metrics with current fixed image and mask image
            self._update_metric(fixed_image_name) 

            # Set current Ground truth pose for current fixed image
            if GroundTruthPose:

                GroundTruthFile =  self.GroundTruthDir + 'ready\\gt_' + self.PatientID + '_' + fixed_image_name + '_' + self.Part + '.csv'
                gt, m = self._initialize_GroundTruth(GroundTruthFile, fixed_image_name)

                print(' Ground Truth pose: ', gt)

            # Initialize landscape for current fixed image
            self.parameters_ranges = self._initialize_landscape(landscape_dim, domain_parameters, GroundTruthPose)

            # Compute 1D landscapes for each of the parameters
            parameters_list = ['Rotx','Roty','Rotz','Translx','Transly','Translz']
            parameter_index = 0

            # Compute 1D landscapes for one parameter only
            #parameters_list = ['Roty']
            #parameter_index = 1

            for which_parameter in parameters_list:
                which_parameter_range = self.parameters_ranges[parameter_index].copy(order = 'C')

                # Initialize vector for parameters (at the centres)
                current_parameters = self.centers.copy(order = 'C') # important to get a copy, otherwise self.centres will be the same as self.parameters

                # checks that Nsamples for the specified parameter is different than 1 (in which case the code does not work well as dimensions get confused)
                assert (len(which_parameter_range) > 1), 'The range for this parameter has only one single value'

                print('\n Evaluation for', which_parameter)

                iter_param_range = 0
                for p in which_parameter_range:
            
                    # Update parameter of interest only
                    current_parameters[parameter_index] = p 

                    #print('Param', current_parameters) 

                    # Compute metrics
                    metric_values = self._compute_metric(current_parameters, save_DRR = save_DRRs, save_id = which_parameter + '_' + str(iter_param_range))

                    # Update 1 dim cost function
                    self.cost_function_1dim[which_parameter][iter_param_range,0] = p
                    self.cost_function_1dim[which_parameter][iter_param_range,1:] = metric_values

                    iter_param_range = iter_param_range + 1


                # Save the new 1D cost function for the current parameter
                self._save_1dim_landscape(which_parameter)

                # Update parameter index
                parameter_index = parameter_index + 1


    def _run_6dim_landscape(self, domain_parameters, 
                                  FixedImageSubDir,  
                                  GroundTruthPose):

        """
            Computes 6 dim Landscape for one single similarity metric, for one single fixed image, for a given range for each of the parameters
            - domain_parameters = dict {'Translx': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Transly': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Translz': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Rotx': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Roty': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}, 
                                        'Rotz': {'Centre'(by default GroundTruthPose), 'Width', 'NSamples'}}
 
            - for the 6dim Landscape, a 4th argument is needed relative to the one single fixed image in analysis
            - one output directory for each metric (if run multiples times with different metrics)

        """  

        landscape_dim = 6

        fixed_image_path = self.ProjectDir + '\\fixed_images\\' + FixedImageSubDir
        fixed_image_filename = self.CurrentOutputsDir.split(sep = '\\')[-1]

        # Update metrics with current fixed image and mask image
        self._update_metric(fixed_image_path, fixed_image_filename)

        # Set current Ground truth pose for current fixed image
        if GroundTruthPose:
            GroundTruthFile = fixed_image_path + '\\ground_truth_poses\\ready\\gt_' + fixed_image_filename + '.csv'
            gt = self._initialize_GroundTruth(GroundTruthFile, fixed_image_filename)

        # Initialize landscape for current fixed image
        self.parameters_ranges = self._initialize_landscape(landscape_dim, domain_parameters, GroundTruthPose)

        iter = 0
        metric = self.Metrics[0]

        print('-----------------------', '\n Computation of 6dim Landscape for Fixed Image: ', fixed_image_filename, ' , Metric: ', metric.info['Name'], '\n rotx, roty, rotz, tx, ty, tz, metric value' )

        for rx in self.range_rotx:
            for ry in self.range_roty:
                for rz in self.range_rotz:
                    for tx in self.range_tx:
                        for ty in self.range_ty:
                            for tz in self.range_tz:

                                # Get set of current parameters
                                current_parameters = [rx, ry, rz, tx, ty, tz]

                                # Compute metric for the current set of parameters
                                metric_value = self._compute_metric(current_parameters)[0]

                                # Update cost function
                                self.cost_function_6dim[iter, 0] = rx
                                self.cost_function_6dim[iter, 1] = ry
                                self.cost_function_6dim[iter, 2] = rz
                                self.cost_function_6dim[iter, 3] = tx
                                self.cost_function_6dim[iter, 4] = ty
                                self.cost_function_6dim[iter, 5] = tz
                                self.cost_function_6dim[iter, 6] = metric_value

                                #print(rx, ry, rz, tx, ty, tz, metric_value )
    
                                iter = iter + 1

        # Save the new 6D cost function
        self._save_6dim_landscape()

        # Compute minimum and save the corresponding DRR
        self._compute_minimum_6dim_landscape()
      

    def _run_optimization(self, initial_parameters,
                                FixedImageSubDir, 
                                GroundTruthPose,
                                PoseInitialization):

        """
            Runs the optimization with one optimizer, for one similarity metric, for all fixed images in the sub-directory FixedImageSubDir, 
            for the given initial parameters (as offset from Groun Truth if GroundTruthPose = True)

        """  

        print('-----------------------', '\n Optimization ', self.Optimizer[0].Name)

        # Loads all fixed images in the specified sub-directory for fixed images.
        fixed_image_path = self.ProjectDir + '\\fixed_images\\' + FixedImageSubDir

        for fixed_image_file in os.listdir(fixed_image_path):
            if fixed_image_file.endswith(self.fixedImgExt):  

                fixed_image_filename = os.path.splitext(fixed_image_file)[0]
                print('\n Fixed Image: ', fixed_image_filename)

                # Overwrite OutputsDir with sub-folder relative to current fixed image
                self.CurrentOutputsDir = self.OutputsDir + '\\' + fixed_image_filename
                if not os.path.isdir(self.CurrentOutputsDir):
                    os.makedirs(self.CurrentOutputsDir)
                if not os.path.isdir(self.CurrentOutputsDir + '\\DRRs'):
                    os.makedirs(self.CurrentOutputsDir + '\\DRRs')  # DRRs folder
                if not os.path.isdir(self.CurrentOutputsDir + '\\Metrics'):
                    os.makedirs(self.CurrentOutputsDir + '\\Metrics') # Metrics folder
               
                # Update metrics with current fixed image and mask image
                self._update_metric(fixed_image_path, fixed_image_filename) 

                # Set current Ground truth pose and optimization parameters for current fixed image
                if GroundTruthPose and not PoseInitialization:

                    GroundTruthFile = fixed_image_path + '\\ground_truth_poses\\ready\\gt_' + fixed_image_filename + '.csv'
                    gt, m = self._initialize_GroundTruth(GroundTruthFile, fixed_image_filename)

                    print(' Ground Truth pose: ', gt)

                    # Update initial guess for optimization parameters as offset from Ground Truth
                    self._opt_parameters = np.add(initial_parameters, gt)

                elif GroundTruthPose and PoseInitialization:

                    GroundTruthFile = fixed_image_path + '\\ground_truth_poses\\ready\\gt_' + fixed_image_filename + '.csv'
                    gt, m = self._initialize_GroundTruth(GroundTruthFile, fixed_image_filename)

                    print(' Ground Truth pose: ', gt)

                    PoseInitFile = fixed_image_path + '\\initialization_poses\\ready\\init_' + fixed_image_filename + '.csv'
                    init = self._initialize_Pose(PoseInitFile, fixed_image_filename)

                    print(' Initialized pose: ', init)

                    # Update initial guess for optimization parameters as offset from initialized position
                    self._opt_parameters = np.add(initial_parameters, init)

                else:

                    # Update initial guess for optimization parameters directly from the provided initial parameters
                    self._opt_parameters = initial_parameters

                # Copy initial guess
                self._initial_guess = np.copy(self._opt_parameters)
                               
                # Update optimizer (only first one) with domain for current fixed image
                self._lowlim, self._uplim = self.Optimizer[0]._set_bound_constraints(self._opt_parameters)

                # Save DRR with initial guess
                self._generateDRR(self._opt_parameters, save_DRR = True, save_id = fixed_image_filename + '_InitialGuess')                

                print('\n Running Optimization... ')

                start_time = time.time()

                # Run optimizer
                # Optimizer always takes real-valued parameters.
                self.minimum_cost_function, self.optimal_parameters = self.Optimizer[0]._optimize(self._opt_parameters, verbose = False)

                elapsed_time = time.time() - start_time

                # Generate and save DRR with found solution
                self._generateDRR(self.optimal_parameters, save_DRR = True, save_id = fixed_image_filename + '_Optimizer_' + self.Optimizer[0].Name)

                # Save history (if required)
                if self._verbose_opt:

                    # Column of row titles
                    hdr = 'MetricValue, Optimized Parameters'

                    # Save history
                    np.savetxt(self.CurrentOutputsDir + '\\Metrics\\OptHistory.csv', self._opt_history , delimiter = ',', newline='\n', header = hdr)

                # Save results as txt file
                file = open(self.CurrentOutputsDir + '\\Metrics\\OptOutcomes.txt','w')
                file.write('Ground Truth Pose: ' + str(gt) + '\n')                 
                for i in range(len(self.Optimizer[0].to_optimize)):
                    file.write('Bound constraints for ' + self.Optimizer[0].param_sequence[self.Optimizer[0].to_optimize[i]] + ': ' + str(self.Optimizer[0].real_lb[i]) + ' , ' + str(self.Optimizer[0].real_ub[i]) + '\n')                    
                file.write('Initial Guess: ' + str(self._initial_guess) + '\n') 
                file.write('Time elapsed: ' + str(elapsed_time) + '\n') 
                file.write('Found optimal pose: '+ str(self.optimal_parameters) + '\n')
                file.write('Found minimum: ' + str(self.minimum_cost_function) + '\n') 
                if GroundTruthPose:
                    file.write('Ground Truth metric: ' + str(m) + '\n')
                    error = self.optimal_parameters - gt
                    error[0:3] = np.rad2deg(error[0:3])
                    file.write('Registration error wrt Ground Truth Pose [deg]' + str(error) + '\n')
 
                file.close()

                # Save results as python dictionary
                results_ = {'GroundTruth':gt, 'OptParam': self.Optimizer[0].to_optimize, \
                            'LowerBounds': self.Optimizer[0].real_lb, 'UpperBounds': self.Optimizer[0].real_ub, \
                            'FoundPose': self.optimal_parameters, 'FoundMin': self.minimum_cost_function}
                if GroundTruthPose:
                    results_['GroundTruthMin'] = m
                np.save(self.CurrentOutputsDir + '\\Metrics\\OptOutcomes.npy', results_)

                # Write Joint Track file of found solution
                file = open(self.CurrentOutputsDir + '\\Metrics\\JointTrackSolution.jts','w')                  
                file.write('JT_EULER_312\n')              
                file.write('          x_tran,          y_tran,          z_tran,                   z_rot,           x_rot,           y_rot\n')
                solution_string = str(self.optimal_parameters[3]) + ',' + str(self.optimal_parameters[4]) + \
                                 ',' + str(self.optimal_parameters[5]) + ',' + str(np.rad2deg(self.optimal_parameters[2])) + ',' + \
                                 str(np.rad2deg(self.optimal_parameters[0])) + ',' + str(np.rad2deg(self.optimal_parameters[1]))
                                    
                file.write(solution_string + '\n')       
                file.close()

                # Print out results
                print(' Optimization finished. Time elapsed: ', elapsed_time)
                print(' Found optimal pose: ', self.optimal_parameters)
                print(' Found minimum: ', self.minimum_cost_function)
                if GroundTruthPose:
                    print(' Registration error wrt Ground Truth Pose [deg]', error)



    def register_cup(self, InitialGuess = 'ViconInit', parameters_offset = np.array([np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), 0.0, 0.0, 0.0], dtype = np.float64)):

        """
            Runs the optimization for cup with one optimizer, for one similarity metric, for all fixed images, 
            with initial guess as input_transform_parameters offset from Ground Truth or from Vicon Initialized pose )

        """  

        # Update for Cup
        self.Part = 'Cup'
        self.Projector = self.ProjectorCup
        self.Optimizer = self.OptimizerCup
        self.CurrentOutputsDir = self.OptHistoryDir               
        JTsolution_file = self.RegistrationDir + self.PatientID + '_' + self.TrialName + '_JTreg_R' + self.Part + '.jts'                
        if not os.path.isdir(self.CurrentOutputsDir + '\\DRRs'):
            os.makedirs(self.CurrentOutputsDir + '\\DRRs')  # DRRs folder
        if not os.path.isdir(self.CurrentOutputsDir + '\\Metrics'):
            os.makedirs(self.CurrentOutputsDir + '\\Metrics') # Metrics folder

        if InitialGuess == 'StemInit':
            init_stem = np.loadtxt(self.RegistrationDir + self.PatientID + '_' + self.TrialName + '_JTreg_RStem.jts', delimiter = ',',skiprows = 2)

        # Load Optimization history file for Cup
        OptHistoryFileCup = self.OptHistoryDir + 'OptHistory_' + self.TrialName + '_' + 'Cup.txt'
        if os.path.exists(OptHistoryFileCup):
            OptHistory = np.loadtxt(OptHistoryFileCup, delimiter = ',', skiprows = 1, dtype=int)
        else:
            sys.exit('Optimization History File for Cup does not exist')
        self.OptHistoryCup['Img#'] = OptHistory[:,0]
        self.OptHistoryCup['To_register'] = OptHistory[:,1]
        self.OptHistoryCup['Pre-Processed'] = OptHistory[:,2]
        self.OptHistoryCup['Registered'] = OptHistory[:,3]
        self.OptHistoryCup['To_correct'] = OptHistory[:,4]

        # Go through all images of the specified trialName 
        img_counter = 0      
        for fixed_img_filepath in self.fixed_images:

            # exit if image is to be registered but it's not been pre-processed
            if self.OptHistoryCup['To_register'][img_counter] and not(self.OptHistoryCup['Pre-Processed'][img_counter]):
                sys.exit('Image needs to be pre-processed first!')

            # if the image needs to be registered, it's not been registered yet or needs to be corrected
            to_register = self.OptHistoryCup['To_register'][img_counter] and not(self.OptHistoryCup['Registered'][img_counter])
            to_correct = self.OptHistoryCup['To_register'][img_counter] and self.OptHistoryCup['To_correct'][img_counter]
            registration_flag = to_register or to_correct
            if registration_flag:

                fixed_img_filename = fixed_img_filepath.split('\\')[-1].split('.')[0]
                print('\n ', fixed_img_filename,  '\n')

                # Update metrics with current fixed image and mask image
                self._update_metric(fixed_img_filename)  

                # Set Initial Guess
                if InitialGuess == 'GroundTruth':

                    GroundTruthFile =  self.GroundTruthDir + 'ready\\gt_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Part + '.csv'
                    gt, m = self._initialize_GroundTruth(GroundTruthFile, fixed_img_filename)

                    print(' Ground Truth pose: ', gt)

                    # Update initial guess for optimization parameters as offset from Ground Truth
                    self._opt_parameters = np.add(parameters_offset, gt)

                elif InitialGuess == 'ViconInit': 

                    PoseInitFile = self.InitPosesDir + 'ready\\init_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Part + '.csv'
                    init = self._initialize_Pose(PoseInitFile, fixed_img_filename)

                    print(' Initialized pose: ', init)

                    # Update initial guess for optimization parameters as offset from initialized position
                    self._opt_parameters = np.add(parameters_offset, init)

                elif InitialGuess == 'StemInit': 

                    # Initialize rotations from Vicon initialization file
                    PoseInitFile = self.InitPosesDir + 'ready\\init_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Part + '.csv'
                    # Load initialization pose
                    init_transform = np.loadtxt(PoseInitFile, delimiter = ';')                    
                    # Get ZXY Euler
                    Euler_zxy_g, e2 = rm.get_euler_zxy(init_transform[:3, :3]) 
                    rotx_init = np.deg2rad(Euler_zxy_g[1])
                    roty_init = np.deg2rad(Euler_zxy_g[2])
                    rotz_init = np.deg2rad(Euler_zxy_g[0])
 
                    # Initialize translations from previously registered stem
                    tx_stem = init_stem[img_counter][0]
                    ty_stem = init_stem[img_counter][1]
                    tz_stem = init_stem[img_counter][2]

                    init = np.array([rotx_init, roty_init, rotz_init, tx_stem, ty_stem, tz_stem])
                    # Update initial guess for optimization parameters as offset from initialized position
                    self._opt_parameters = np.add(parameters_offset, init)
                                            
                # Copy initial guess
                initial_guess = np.copy(self._opt_parameters)
                               
                # Update optimizer with domain for current fixed image
                self._lowlim, self._uplim = self.Optimizer._set_bound_constraints(self._opt_parameters)

                # Save DRR with initial guess
                self._generateDRR(self._opt_parameters, save_DRR = True, save_id = fixed_img_filename + '_InitialGuess_' + self.Part)                

                print('\n Running Optimization for ', fixed_img_filename)

                start_time = time.time()

                # Run optimizer
                # Optimizer always takes real-valued parameters.
                minimum_cost_function, optimal_parameters = self.Optimizer._optimize(self._opt_parameters, verbose = False)
                #minimum_cost_function = 0.
                #optimal_parameters = self._opt_parameters

                elapsed_time = time.time() - start_time

                # Generate and save DRR with found solution
                self._generateDRR(optimal_parameters, save_DRR = True, save_id = fixed_img_filename + '_' + self.Part + '_Optimizer_' + self.Optimizer.Name)

                # Save results as python dictionary
                results_ = {'OptParam': self.Optimizer.to_optimize, \
                            'LowerBounds': self.Optimizer.real_lb, 'UpperBounds': self.Optimizer.real_ub, \
                            'FoundPose': optimal_parameters, 'FoundMin': minimum_cost_function}
                if InitialGuess == 'GroundTruth':
                    results_['GroundTruthMin'] = m
                np.save(self.CurrentOutputsDir + 'OptOutcomes_' + fixed_img_filename + '_' + self.Part +'.npy', results_)
        
                # Save results as txt file
                file = open(self.CurrentOutputsDir + 'OptOutcomes_' + fixed_img_filename + '_' + self.Part +'.txt','w')                 
                for i in range(len(self.Optimizer.to_optimize)):
                    file.write('Bound constraints for ' + self.Optimizer.param_sequence[self.Optimizer.to_optimize[i]] + ': ' + str(self.Optimizer.real_lb[i]) + ' , ' + str(self.Optimizer.real_ub[i]) + '\n')                    
                file.write('Initial Guess: ' + str(initial_guess) + '\n') 
                file.write('Time elapsed: ' + str(elapsed_time) + '\n') 
                file.write('Found optimal pose: '+ str(optimal_parameters) + '\n')
                file.write('Found minimum: ' + str(minimum_cost_function) + '\n') 
                if InitialGuess == 'GroundTruth':
                    file.write('Ground Truth metric: ' + str(m) + '\n')
                    error = optimal_parameters - gt
                    error[0:3] = np.rad2deg(error[0:3])
                    file.write('Registration error wrt Ground Truth Pose [deg]' + str(error) + '\n')
                file.close()

                # Update Joint Track file with found solution
                with open(JTsolution_file, 'r') as JTfile:
                    JTdata = JTfile.readlines()
                JTdata[img_counter+2] = str(optimal_parameters[3]) + ',' + str(optimal_parameters[4]) + \
                                 ',' + str(optimal_parameters[5]) + ',' + str(np.rad2deg(optimal_parameters[2])) + ',' + \
                                 str(np.rad2deg(optimal_parameters[0])) + ',' + str(np.rad2deg(optimal_parameters[1])) + '\n'
                with open(JTsolution_file, 'w') as JTfile:
                    JTfile.writelines( JTdata )

                # Update and save new Optimization history (back-up for crashes)
                self.OptHistoryCup['Registered'][img_counter] = 1
                #self.OptHistoryCup['To_correct'][img_counter] = 1 # Temporary
                file = open(OptHistoryFileCup,'w')                              
                file.write('Img#, To_register, Pre-Processed, Registered, To_correct\n')
                for s in range(self.num_fixed_images):
                    s_h = str(str(self.OptHistoryCup['Img#'][s]) + ',' + str(self.OptHistoryCup['To_register'][s]) + ',' + str(self.OptHistoryCup['Pre-Processed'][s]) + ',' + str(self.OptHistoryCup['Registered'][s]) + ',' + str(self.OptHistoryCup['To_correct'][s]))
                    file.write(s_h + '\n')       
                file.close()

             # Update Fixed Image counter
            img_counter = img_counter + 1
                         
        print('-----------------------', '\n Optimization ', self.Optimizer.Name)


    def register_stem(self, InitialGuess = 'ViconInit', parameters_offset = np.array([np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), 0.0, 0.0, 0.0], dtype = np.float64)):

        """
            Runs the optimization for cup with one optimizer, for one similarity metric, for all fixed images, 
            with initial guess as input_transform_parameters offset from Ground Truth or from Vicon Initialized pose )

        """  

        # Update for Stem
        self.Part = 'Stem'
        self.Projector = self.ProjectorStem
        self.Optimizer = self.OptimizerStem
        self.CurrentOutputsDir = self.OptHistoryDir               
        JTsolution_file = self.RegistrationDir + self.PatientID + '_' + self.TrialName + '_JTreg_R' + self.Part + '.jts'                
        if not os.path.isdir(self.CurrentOutputsDir + '\\DRRs'):
            os.makedirs(self.CurrentOutputsDir + '\\DRRs')  # DRRs folder
        if not os.path.isdir(self.CurrentOutputsDir + '\\Metrics'):
            os.makedirs(self.CurrentOutputsDir + '\\Metrics') # Metrics folder
        
        if InitialGuess == 'ViconInitAndCupTransl':
            init_cup = np.loadtxt(self.RegistrationDir + self.PatientID + '_' + self.TrialName + '_JTreg_RCup.jts', delimiter = ',',skiprows = 2)

        # Load Optimization history file for Stem
        OptHistoryFileStem = self.OptHistoryDir + 'OptHistory_' + self.TrialName + '_' + 'Stem.txt'
        if os.path.exists(OptHistoryFileStem):
            OptHistory = np.loadtxt(OptHistoryFileStem, delimiter = ',', skiprows = 1, dtype=int)
        else:
            sys.exit('Optimization History File for Stem does not exist!')
        self.OptHistoryStem['Img#'] = OptHistory[:,0]
        self.OptHistoryStem['To_register'] = OptHistory[:,1]
        self.OptHistoryStem['Pre-Processed'] = OptHistory[:,2]
        self.OptHistoryStem['Registered'] = OptHistory[:,3]
        self.OptHistoryStem['To_correct'] = OptHistory[:,4]

        # Go through all images of the specified trialName 
        img_counter = 0       
        for fixed_img_filepath in self.fixed_images:

            # exit if image is to be registered but it's not been pre-processed
            if self.OptHistoryStem['To_register'][img_counter] and not(self.OptHistoryStem['Pre-Processed'][img_counter]):
                sys.exit('Image needs to be pre-processed first!')

            # if the image needs to be registered, it's not been registered yet or needs to be corrected
            to_register = self.OptHistoryStem['To_register'][img_counter] and not(self.OptHistoryStem['Registered'][img_counter])
            to_correct = self.OptHistoryStem['To_register'][img_counter] and self.OptHistoryStem['To_correct'][img_counter]
            registration_flag = to_register or to_correct
            if registration_flag:

                fixed_img_filename = fixed_img_filepath.split('\\')[-1].split('.')[0]
                print('\n ', fixed_img_filename,  '\n')

                # Update metrics with current fixed image and mask image
                self._update_metric(fixed_img_filename)  

                # Set Initial Guess
                if InitialGuess == 'GroundTruth':

                    GroundTruthFile =  self.GroundTruthDir + 'ready\\gt_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Part + '.csv'
                    gt, m = self._initialize_GroundTruth(GroundTruthFile, fixed_img_filename)

                    print(' Ground Truth pose: ', gt)

                    # Update initial guess for optimization parameters as offset from Ground Truth
                    self._opt_parameters = np.add(parameters_offset, gt)

                elif InitialGuess == 'ViconInit': 

                    PoseInitFile = self.InitPosesDir + 'ready\\init_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Part + '.csv'
                    init = self._initialize_Pose(PoseInitFile, fixed_img_filename)

                    print(' Initialized pose: ', init)

                    # Update initial guess for optimization parameters as offset from initialized position
                    self._opt_parameters = np.add(parameters_offset, init)

                elif InitialGuess == 'ViconInitAndCupTransl':

                    # Initialize rotations from Vicon initialization file
                    PoseInitFile = self.InitPosesDir + 'ready\\init_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Part + '.csv'
                    # Load initialization pose
                    init_transform = np.loadtxt(PoseInitFile, delimiter = ';')                    
                    # Get ZXY Euler
                    Euler_zxy_g, e2 = rm.get_euler_zxy(init_transform[:3, :3]) 
                    rotx_init = np.deg2rad(Euler_zxy_g[1])
                    roty_init = np.deg2rad(Euler_zxy_g[2])
                    rotz_init = np.deg2rad(Euler_zxy_g[0])

                    # Initialize translations from previously registered cup
                    #init_cup = np.load(self.CurrentOutputsDir + 'OptOutcomes_' + fixed_img_filename + '_Cup.npy').item()                    
                    tx_cup = init_cup[img_counter][0]
                    ty_cup = init_cup[img_counter][1]
                    tz_cup = init_cup[img_counter][2]

                    init = np.array([rotx_init, roty_init, rotz_init, tx_cup, ty_cup, tz_cup])
                    # Update initial guess for optimization parameters as offset from initialized position
                    self._opt_parameters = np.add(parameters_offset, init)

                         
                # Copy initial guess
                initial_guess = np.copy(self._opt_parameters)
                               
                # Update optimizer with domain for current fixed image
                self._lowlim, self._uplim = self.Optimizer._set_bound_constraints(self._opt_parameters)

                # Save DRR with initial guess
                self._generateDRR(self._opt_parameters, save_DRR = True, save_id = fixed_img_filename + '_InitialGuess_' + self.Part)                

                print('\n Running Optimization for ', fixed_img_filename)

                start_time = time.time()

                # Run optimizer
                # Optimizer always takes real-valued parameters.
                minimum_cost_function, optimal_parameters = self.Optimizer._optimize(self._opt_parameters, verbose = False)
                #minimum_cost_function = 0.
                #optimal_parameters = self._opt_parameters

                elapsed_time = time.time() - start_time

                # Generate and save DRR with found solution
                self._generateDRR(optimal_parameters, save_DRR = True, save_id = fixed_img_filename + '_' + self.Part + '_Optimizer_' + self.Optimizer.Name)
        
                # Save results as txt file
                file = open(self.CurrentOutputsDir + 'OptOutcomes_' + fixed_img_filename + '_' + self.Part +'.txt','w')                 
                for i in range(len(self.Optimizer.to_optimize)):
                    file.write('Bound constraints for ' + self.Optimizer.param_sequence[self.Optimizer.to_optimize[i]] + ': ' + str(self.Optimizer.real_lb[i]) + ' , ' + str(self.Optimizer.real_ub[i]) + '\n')                    
                file.write('Initial Guess: ' + str(initial_guess) + '\n') 
                file.write('Time elapsed: ' + str(elapsed_time) + '\n') 
                file.write('Found optimal pose: '+ str(optimal_parameters) + '\n')
                file.write('Found minimum: ' + str(minimum_cost_function) + '\n') 
                if InitialGuess == 'GroundTruth':
                    file.write('Ground Truth metric: ' + str(m) + '\n')
                    error = optimal_parameters - gt
                    error[0:3] = np.rad2deg(error[0:3])
                    file.write('Registration error wrt Ground Truth Pose [deg]' + str(error) + '\n')
                file.close()

                # Update Joint Track file with found solution
                with open(JTsolution_file, 'r') as JTfile:
                    JTdata = JTfile.readlines()
                JTdata[img_counter+2] = str(optimal_parameters[3]) + ',' + str(optimal_parameters[4]) + \
                                 ',' + str(optimal_parameters[5]) + ',' + str(np.rad2deg(optimal_parameters[2])) + ',' + \
                                 str(np.rad2deg(optimal_parameters[0])) + ',' + str(np.rad2deg(optimal_parameters[1])) + '\n'
                with open(JTsolution_file, 'w') as JTfile:
                    JTfile.writelines( JTdata )

                # Update and save new Optimization history (back-up for crashes)
                self.OptHistoryStem['Registered'][img_counter] = 1
                #self.OptHistoryStem['To_correct'][img_counter] = 1 # Temporary
                file = open(OptHistoryFileStem,'w')                              
                file.write('Img#, To_register, Pre-Processed, Registered, To_correct\n')
                for s in range(self.num_fixed_images):
                    s_h = str(str(self.OptHistoryStem['Img#'][s]) + ',' + str(self.OptHistoryStem['To_register'][s]) + ',' + str(self.OptHistoryStem['Pre-Processed'][s]) + ',' + str(self.OptHistoryStem['Registered'][s]) + ',' + str(self.OptHistoryStem['To_correct'][s]))
                    file.write(s_h + '\n')       
                file.close() 

             # Update Fixed Image counter
            img_counter = img_counter + 1
                         
        print('-----------------------', '\n Optimization ', self.Optimizer.Name)


    def register_Pelvis(self, InitialGuess = 'ViconInit', parameters_offset = np.array([np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), 0.0, 0.0, 0.0], dtype = np.float64)):

        """
            Runs the optimization for cup with one optimizer, for one similarity metric, for all fixed images, 
            with initial guess as input_transform_parameters offset from Ground Truth or from Vicon Initialized pose )

        """  

        # Update for Stem
        self.Part = 'Pelvis'
        self.Projector = self.ProjectorPelvis
        self.Optimizer = self.OptimizerPelvis
        self.CurrentOutputsDir = self.OptHistoryDir               
        if not os.path.isdir(self.CurrentOutputsDir + '\\DRRs'):
            os.makedirs(self.CurrentOutputsDir + '\\DRRs')  # DRRs folder
        if not os.path.isdir(self.CurrentOutputsDir + '\\Metrics'):
            os.makedirs(self.CurrentOutputsDir + '\\Metrics') # Metrics folder
        
        #if InitialGuess == 'ViconInit':
        #    init_cup = np.loadtxt(self.RegistrationDir + self.PatientID + '_' + self.TrialName + '_JTreg_RCup.jts', delimiter = ',',skiprows = 2)

        # Go through all images of the specified trialName 
        img_counter = 0       
        for fixed_img_filepath in self.fixed_images:

            fixed_img_filename = fixed_img_filepath.split('\\')[-1].split('.')[0]
            print('\n ', fixed_img_filename,  '\n')

            # Update metrics with current fixed image and mask image
            self._update_metric(fixed_img_filename)  

            # Set Initial Guess
            if InitialGuess == 'GroundTruth':

                GroundTruthFile =  self.GroundTruthDir + 'ready\\gt_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Part + '.csv'
                gt, m = self._initialize_GroundTruth(GroundTruthFile, fixed_img_filename)

                print(' Ground Truth pose: ', gt)

                # Update initial guess for optimization parameters as offset from Ground Truth
                self._opt_parameters = np.add(parameters_offset, gt)

            elif InitialGuess == 'ViconInit': 

                PoseInitFile = self.InitPosesDir + 'ready\\init_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Part + '.csv'
                init = self._initialize_Pose(PoseInitFile, fixed_img_filename)

                print(' Initialized pose: ', init)

                # Update initial guess for optimization parameters as offset from initialized position
                self._opt_parameters = np.add(parameters_offset, init)
                         
             # Copy initial guess
            initial_guess = np.copy(self._opt_parameters)
                               
            # Update optimizer with domain for current fixed image
            self._lowlim, self._uplim = self.Optimizer._set_bound_constraints(self._opt_parameters)

            # Save DRR with initial guess
            self._generateDRR(self._opt_parameters, save_DRR = True, save_id = fixed_img_filename + '_InitialGuess_' + self.Part)                

            print('\n Running Optimization for ', fixed_img_filename)

            start_time = time.time()

            # Run optimizer
            # Optimizer always takes real-valued parameters.
            minimum_cost_function, optimal_parameters = self.Optimizer._optimize(self._opt_parameters, verbose = False)
            #minimum_cost_function = 0.
            #optimal_parameters = self._opt_parameters

            elapsed_time = time.time() - start_time

            # Generate and save DRR with found solution
            self._generateDRR(optimal_parameters, save_DRR = True, save_id = fixed_img_filename + '_' + self.Part + '_Optimizer_' + self.Optimizer.Name)
        
            # Save results as txt file
            file = open(self.CurrentOutputsDir + 'OptOutcomes_' + fixed_img_filename + '_' + self.Part +'.txt','w')                 
            for i in range(len(self.Optimizer.to_optimize)):
                file.write('Bound constraints for ' + self.Optimizer.param_sequence[self.Optimizer.to_optimize[i]] + ': ' + str(self.Optimizer.real_lb[i]) + ' , ' + str(self.Optimizer.real_ub[i]) + '\n')                    
            file.write('Initial Guess: ' + str(initial_guess) + '\n') 
            file.write('Time elapsed: ' + str(elapsed_time) + '\n') 
            file.write('Found optimal pose: '+ str(optimal_parameters) + '\n')
            file.write('Found minimum: ' + str(minimum_cost_function) + '\n') 
            if InitialGuess == 'GroundTruth':
                file.write('Ground Truth metric: ' + str(m) + '\n')
                error = optimal_parameters - gt
                error[0:3] = np.rad2deg(error[0:3])
                file.write('Registration error wrt Ground Truth Pose [deg]' + str(error) + '\n')
            file.close()

            # Update and save new Optimization history (back-up for crashes)
            self.OptHistoryStem['Registered'][img_counter] = 1
            #self.OptHistoryStem['To_correct'][img_counter] = 1 # Temporary
            file = open(OptHistoryFileStem,'w')                              
            file.write('Img#, To_register, Pre-Processed, Registered, To_correct\n')
            for s in range(self.num_fixed_images):
                s_h = str(str(self.OptHistoryStem['Img#'][s]) + ',' + str(self.OptHistoryStem['To_register'][s]) + ',' + str(self.OptHistoryStem['Pre-Processed'][s]) + ',' + str(self.OptHistoryStem['Registered'][s]) + ',' + str(self.OptHistoryStem['To_correct'][s]))
                file.write(s_h + '\n')       
            file.close() 

             # Update Fixed Image counter
            img_counter = img_counter + 1
                         
        print('-----------------------', '\n Optimization ', self.Optimizer.Name)



    def register(self, InitialGuess = 'fromUser', 
                       parameters_offset = np.array([np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), 0.0, 0.0, 0.0], dtype = np.float64)):

        """ Runs 2D/3D registration for all images, based on the (first) specified similarity metric, 
            and based on the specified initial guess (plus a optional offset for each pose parameter)
        """  

        # Define output directories
        self.CurrentOutputsDir = self.OptHistoryDir               
        if not os.path.isdir(self.CurrentOutputsDir + '\\DRRs'):
            os.makedirs(self.CurrentOutputsDir + '\\DRRs')      # folder where DRRs will be saved
        if not os.path.isdir(self.CurrentOutputsDir + '\\Metrics'):
            os.makedirs(self.CurrentOutputsDir + '\\Metrics')       # folder where metric values will be stored
        
        #if InitialGuess == 'ViconInit':
        #    init_cup = np.loadtxt(self.RegistrationDir + self.PatientID + '_' + self.TrialName + '_JTreg_RCup.jts', delimiter = ',',skiprows = 2)

        # Go through all images of the specified trialName 
        img_counter = 0       
        for fixed_img_filepath in self.fixed_images:

            fixed_img_filename = fixed_img_filepath.split('\\')[-1].split('.')[0]
            print('\n ', fixed_img_filename,  '\n')

            # Update metrics with current fixed image and mask image
            self._update_metric(fixed_img_filename)  

            # Set Initial Guess
            if InitialGuess == 'GroundTruth':

                GroundTruthFile =  self.GroundTruthDir + 'ready\\gt_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Model3D + '.csv'
                gt, m = self._initialize_GroundTruth(GroundTruthFile, fixed_img_filename)

                print(' Ground Truth pose: ', gt)

                # Update initial guess for optimization parameters as offset from Ground Truth
                self._opt_parameters = np.add(parameters_offset, gt)

            elif InitialGuess == 'fromUser': 

                PoseInitFile = self.InitPosesDir + 'init_' + self.PatientID + '_' + fixed_img_filename + '_' + self.Model3D + '.csv'
                init = self._initialize_Pose(PoseInitFile, fixed_img_filename)

                print(' Initialized pose: ', init)

                # Update initial guess for optimization parameters as offset from initialized position
                self._opt_parameters = np.add(parameters_offset, init)
                         
             # Copy initial guess
            initial_guess = np.copy(self._opt_parameters)
                               
            # Update optimizer with search domain for current fixed image
            self._lowlim, self._uplim = self.Optimizer._set_bound_constraints(self._opt_parameters)

            # Save DRR with initial guess
            self._generateDRR(self._opt_parameters, save_DRR = True, save_id = fixed_img_filename + '_InitialGuess_' + self.Model3D)                

            print('\n Running Optimization for ', fixed_img_filename)

            start_time = time.time()

            # Run optimizer
            minimum_cost_function, optimal_parameters = self.Optimizer._optimize(self._opt_parameters, verbose = False)

            elapsed_time = time.time() - start_time

            # Generate and save DRR with found solution
            self._generateDRR(optimal_parameters, save_DRR = True, save_id = fixed_img_filename + '_' + self.Model3D + '_Optimizer_' + self.Optimizer.Name)
        
            # Save results as txt file
            file = open(self.CurrentOutputsDir + 'OptOutcomes_' + fixed_img_filename + '_' + self.Model3D +'.txt','w')                 
            for i in range(len(self.Optimizer.to_optimize)):
                file.write('Bound constraints for ' + self.Optimizer.param_sequence[self.Optimizer.to_optimize[i]] + ': ' + str(self.Optimizer.real_lb[i]) + ' , ' + str(self.Optimizer.real_ub[i]) + '\n')                    
            file.write('Initial Guess: ' + str(initial_guess) + '\n') 
            file.write('Time elapsed: ' + str(elapsed_time) + '\n') 
            file.write('Found optimal pose: '+ str(optimal_parameters) + '\n')
            file.write('Found minimum: ' + str(minimum_cost_function) + '\n') 
            if InitialGuess == 'GroundTruth':
                file.write('Ground Truth metric: ' + str(m) + '\n')
                error = optimal_parameters - gt
                error[0:3] = np.rad2deg(error[0:3])
                file.write('Registration error wrt Ground Truth Pose [deg]' + str(error) + '\n')
            file.close()

             # Update Fixed Image counter
            img_counter = img_counter + 1

        # Delete HipHop object at the end of the task
        self._delete_me()
                         
        print('-----------------------', '\n Optimization ', self.Optimizer.Name)


    def run(self, taskType,
                  input_transform_parameters, 
                  FixedImages,
                  part = 'Cup', 
                  FixedImgFileName = None, 
                  GroundTruthPose = True,
                  PoseInitialization = False,
                  save_DRRs = False):

        """
            Only function supposed to be called by the user. 
            It loads all fixed images in the specified sub-directory FixedImageSubDir\ready\ of the ProjectDir\fixed_images folder.
            It loads the specified mask images (or all mask images) in the specified sub-directory FixedImageSubDir\masks\ready of the ProjectDir\fixed_images folder.
            It runs on of the following tasks, as specified by the input string taskType:

                - computation of the metric values for a specific case study
                - computation of the 1 dimensional landscape for different metrics
                - computation of the 6 dimensional landscape for one metric
                - optimization for one metric 

            OutputsDir
            --> Case_study or Landscape1dim or Landscape6dim
                --> fixed_image1
                    --> DRRs
                        --- DRRGroundTruthPose.mha (not for Case_study)
                        --- DRR1.mha
                    --> Metrics
                        --- Metrics_GroundTruthPose.txt (not for Case_study)
                        --- Metrics.txt

        """

        # Directories 
        self.OutputsDir = self.RegistrationDir + '\\' + taskType
        self.CurrentOutputsDir = self.OutputsDir

        self.Part = part
        if part == 'Cup':
            self.Projector = self.ProjectorCup
        elif part == 'Stem':
            self.Projector = self.ProjectorStem

        # Case Study
        if taskType == 'Case_study':

            if not os.path.isdir(self.OutputsDir + '\\DRRs'):
                os.makedirs(self.OutputsDir + '\\DRRs')  # DRRs folder
            if not os.path.isdir(self.OutputsDir + '\\Metrics'):
                os.makedirs(self.OutputsDir + '\\Metrics') # Metrics folder

            self._run_case_study(input_transform_parameters, FixedImageSubDir, GroundTruthPose, save_DRRs)


        # 1dim Landscape
        if taskType == 'Landscape1dim':

            if not os.path.isdir(self.OutputsDir):
                os.makedirs(self.OutputsDir)

            self._run_1dim_landscape(input_transform_parameters, FixedImages, GroundTruthPose, save_DRRs)


        # 6dim Landscape
        if taskType == 'Landscape6dim':

            # Check that one single fixed image was specified
            if FixedImgFileName == None: sys.exit('No specific fixed image was selected for 6dim landscape')

            # Directories (make sure not to overwrite already computed landscapes)
            self.CurrentOutputsDir = self.OutputsDir + '\\' + FixedImgFileName
            if os.path.isdir(self.CurrentOutputsDir):
               sys.exit(self.CurrentOutputsDir + ' directory already exists. Do not overwrite!')
            else:
                os.makedirs(self.CurrentOutputsDir)
            os.makedirs(self.CurrentOutputsDir + '\\DRRs')  # DRRs folder
            os.makedirs(self.CurrentOutputsDir + '\\Metrics') # Metrics folder

            self._run_6dim_landscape(input_transform_parameters, FixedImageSubDir, GroundTruthPose)
                

        # Optimization
        if taskType == 'Optimization':

            if not os.path.isdir(self.OutputsDir):
                os.makedirs(self.OutputsDir)
            
            self._run_optimization(input_transform_parameters, FixedImageSubDir, GroundTruthPose, PoseInitialization)


        # Delete HipHop object (and metrics) at the end of the task
        self._delete_me()
        print('----------------------- \n Bye HipHop! \n-----------------------')



        



    