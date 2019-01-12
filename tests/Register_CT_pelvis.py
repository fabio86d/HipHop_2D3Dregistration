"""
    Fabio D'Isidoro - ETH Zurich - November 2018

    A framework to perform 2D/3D registration between an CT or STL model of the hip joint and a radiograph of the pelvis,
    in order to determine the 3D pose of the hip joint.

"""

from sys import argv
from HipHop import *
import time


# Info patient and motion activity
ADL_info = {'PatientID': 'HipPhantom', 
            'ActivityType': 'StaticPose', 
            'TrialName': 'C_18d_3mm'}

# Define similarity metric
metrics_info = [{'Name': 'GradCorr',
                 'NormFixImg': False,
                 'NormMovImg': False,
                 'SubtractMeanOn': False,
                 'saveFixedImages': False,
                 'saveMovImages' : False}]

# Define projector for generation of DRR from 3D model (Digitally Reconstructed Radiographs)
projector_info = {'Name': 'SiddonGpu', 
                  'threadsPerBlock_x': 16,
                  'threadsPerBlock_y': 16,
                  'threadsPerBlock_z': 1,
                  '3Dmodel': 'Pelvis'}

# Optimizer to find best similarity
optimizer_info = {'Library': 'NL_opt', 
                  'Name': 'GN_ESCH',        # Evolutionary Strategy optimizer from NLopt library 
                  'Dim' : ['Rotx', 'Rotz'],     # pose parameters to be optimized
                  'domain_range': [np.deg2rad(20.), np.deg2rad(20.)],       # search range for each pose parameter 
                  'max_eval' : 100,        # max number iterations 
                  'Norm': False,        # normalize pose paramerters and search domain 
                  'Verbose' : True}
                  

# Read inputs
ProjectDir = argv[1]

# Create registration object (instance of class HipHop)
NewRegistration = HipHop(ProjectDir,
                         ADL_info,
                         projector_info, 
                         metrics_info,
                         optimizer_info,
                         GroundTruthPose = True)

# Run 2D/3D registration based on Initial Guess equal to the ground truth pose plus a specified offset for each pose parameter (rotX, rotY, rotZ, translX, translY, translZ)
t1 = time.time()
NewRegistration.register(InitialGuess = 'GroundTruth', parameters_offset = np.array([np.deg2rad(8.0), np.deg2rad(0.0), np.deg2rad(8.0), 0.0, 0.0, 0.0]))
t2 = time.time()

print('Registration time:', t2 -t1)