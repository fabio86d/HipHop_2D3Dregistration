# GPU accelerated 2D/3D registration
# Brief Desription
The repository contains a framework for 2D/3D image registration between CT or MRI scan to and 2D X-ray images.

# Installation
Use the code is with an Anaconda environment equipped with python-3.X, Python itk, Python vtk, Python openCV and Cython.

Procedure:

    1) Create the Anaconda environment (if not created yet): conda create -n HipHop python=3 anaconda.

    2) Install required Python packages: 

	a. Install itk with (https://discourse.itk.org/t/itkvtkglue-module-in-python-wheel/393):
	pip install itk
	
	b. Install vtk with (https://stackoverflow.com/questions/43184009/install-vtk-with-anaconda-3-6 ): 
	conda install -c clinicalgraphics vtk

	c. Install opencv:
	download the unofficial binary python wheels for python 3.6 from https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv,
	and install following the instructions from  https://stackoverflow.com/questions/42994813/installing-opencv-on-windows-10-with-python-3-6-and-anaconda-3-6

	d. Install NLopt library for Python
	(https://nlopt.readthedocs.io/en/latest/NLopt_on_Windows/)

    3) Activate the environment: source activate HipHop.

    4) git clone https://github.com/fabio86d/HipHop_2D3Dregistration.git.


# Test the package

In order to run 2D/3D registration between STL model and an X-ray image:
run python main_implant.py