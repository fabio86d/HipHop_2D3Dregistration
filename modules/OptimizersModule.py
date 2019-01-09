"""Optimizers for 2D-3D image registration.

This module includes different optimizers and an optimizers class factory.

Classes:
    NL_opt_GH_ESCH: Evolutionary Strategy optimizer from NLopt library. 
    NL_opt_LN_NELDERMEAD: Nelder-Mead (Simplex) optimizer from NLopt library.

Functions:
    optimizer_factory: returns an optimizer instance.
    
New Optimizers can be plugged-in and added to the optimizers factory
as long as they are defined as classes with the following methods:
    _set_bound_constraints: sets lower an upper bounds to the optimizer
    _optimize: runs the optimization and returns the minimum values and its
        location in the parameters space. 
"""



####  PYTHON MODULES
import sys
import numpy as np



####  OPTIMIZATION LIBRARY
sys.path.append('C:\\NLopt\\nlopt-2.4.2-dll64_2nd')
import nlopt



def optimizer_factory(optimizer_info, cost_function):

    """Generates instances of the specified optimizer.

    Args:
        optimizer_info (dict of str): dictionary with the following keys:
            - Library: 'NLopt'
            - Name: 'GN_ESCH' or 'LN_NELDERMEAD'
            - Dim: list of strings within 'Rotx', 'Roty','Rotz','TransX','TransY','TransZ'
            - domain_range: list of floats, each corresponding to the half width 
                of the search range for the pose parameters listed in Dim
            - max_eval: int for maximum number of optimizer itarations (convergence criterion)
            - Norm: bool (if True the search domain normalized within [0,1])
        cost_function (function): cost function returning the metric value

    Returns:
        opt: one of the optimizer classes.
    """

    if optimizer_info['Library'] == 'NL_opt' and optimizer_info['Name'] == 'GN_ESCH':

        opt = NL_opt_GH_ESCH(optimizer_info, cost_function)

        return opt

    if optimizer_info['Library'] == 'NL_opt' and optimizer_info['Name'] == 'LN_NELDERMEAD':

        opt = NL_opt_LN_NELDERMEAD(optimizer_info, cost_function)

        return opt



class NL_opt_GH_ESCH():

    """Evolutionary Strategy optimizer from the NLopt library.
    
       https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/

       Methods:
            _set_bound_constraints
            _optimize       
    """

    def __init__(self, optimizer_info, objective_function):

        # Create Optimizer
        self.Dim = len(optimizer_info['Dim'])
        self.Name = optimizer_info['Name']
        self.opt_GN_ESCH = nlopt.opt(nlopt.GN_ESCH, self.Dim)

        # Create binary vector for parameters to be optimized
        self.to_optimize = []
        self.param_sequence = ['Rotx', 'Roty', 'Rotz', 'Translx','Transly', 'Translz']
        for i in range(len(self.param_sequence)):
            if self.param_sequence[i] in optimizer_info['Dim']:
                self.to_optimize.append(i)

        # Set objective function
        self.opt_GN_ESCH.set_min_objective(objective_function)

        # Instantiate lower and upper bounds
        self.Norm = optimizer_info['Norm']

        # Optimizer-specific parameters
        self.DomainRange = optimizer_info['domain_range']
        if optimizer_info['max_eval']:
            self.opt_GN_ESCH.set_maxeval(optimizer_info['max_eval'])


    
    def _set_bound_constraints(self, initial_parameters):

        """Sets lower and upper bounds based on current initial parameters.

            Args:
                initial_parameters: list of floats (initial guess for each pose parameter)

            Return:  
                real_lb, real_ub: floats for lower and upper bound                      
        """

        # Generate real-valued bound constraints for current initial parameters
        self.real_lb = []
        self.real_ub = []

        for i in range(len(self.to_optimize)):
            self.real_lb.append(initial_parameters[self.to_optimize[i]] - self.DomainRange[i]) 
            self.real_ub.append(initial_parameters[self.to_optimize[i]] + self.DomainRange[i])

        for i in range(len(self.to_optimize)):
            print(' \n Bound constraints for ', self.param_sequence[self.to_optimize[i]], ': ', self.real_lb[i], self.real_ub[i])


        if self.Norm:

            # Assign normalized bound constraints
            self.opt_GN_ESCH.set_lower_bounds([0.0]*self.Dim)
            self.opt_GN_ESCH.set_upper_bounds([1.0]*self.Dim)

        else:

            # Assign real bound constraints
            self.opt_GN_ESCH.set_lower_bounds(self.real_lb)
            self.opt_GN_ESCH.set_upper_bounds(self.real_ub)        
        
        return self.real_lb, self.real_ub



    def _optimize(self, parameters, verbose = False):

        """Runs the Evolutionary Strategy optimization

            Args:
                parameters: list of floats (initial guess for each pose parameter)            

            Return:      
                minf (float): found minimum
                parameters (list of floats): minimum location                    
        """

        # Pick initial (real) parameters to optimize only
        parameters_to_optimize = parameters[self.to_optimize]

        #print('\n Initial Guess is: ', parameters_to_optimize)

        # Normalize initial parameters (if needed)
        if self.Norm:
            parameters_to_optimize = list(np.divide(parameters_to_optimize - np.asarray(self.real_lb), (np.asarray(self.real_ub) - np.asarray(self.real_lb))))
    
        # Run optimization
        optimal_parameters = self.opt_GN_ESCH.optimize(parameters_to_optimize)

        # De-normalized optimal parameters (if needed)
        if self.Norm:
            optimal_parameters = list(np.multiply((np.asarray(self.real_ub) - np.asarray(self.real_lb)),optimal_parameters) + np.asarray(self.real_lb))

        # Get optimization results
        minf = self.opt_GN_ESCH.last_optimum_value()

        if verbose:
            print("found minimum at ", optimal_parameters)
            print("minimum value = ", minf)
            print("result code = ", self.opt_GN_ESCH.last_optimize_result())

        # Update final solution
        parameters[self.to_optimize] = optimal_parameters

        return minf, parameters





class NL_opt_LN_NELDERMEAD():

    """Nelder-Mead optimizer from the NLopt library.
    
       https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/

       Methods:
            _set_bound_constraints
            _optimize       
    """

    def __init__(self, optimizer_info, objective_function):

        # Create Optimizer
        self.Dim = len(optimizer_info['Dim'])
        self.opt_LN_NELDERMEAD = nlopt.opt(nlopt.LN_NELDERMEAD, self.Dim)
        self.Name = optimizer_info['Name']

        # Create binary vector for parameters to be optimized
        self.to_optimize = []
        self.param_sequence = ['Rotx', 'Roty', 'Rotz', 'Translx','Transly', 'Translz']
        for i in range(len(self.param_sequence)):
            if self.param_sequence[i] in optimizer_info['Dim']:
                self.to_optimize.append(i)

        # Set objective function
        self.opt_LN_NELDERMEAD.set_min_objective(objective_function)

        # Instantiate lower and upper bounds
        self.Norm = optimizer_info['Norm']

        # Optimizer-specific parameters
        self.DomainRange = optimizer_info['domain_range']

        if optimizer_info['max_eval']:
            self.opt_LN_NELDERMEAD.set_maxeval(optimizer_info['max_eval'])

        if optimizer_info['xtol_rel']:
            self.opt_LN_NELDERMEAD.set_xtol_rel(optimizer_info['xtol_rel'])


    
    def _set_bound_constraints(self, initial_parameters):

        """Sets lower and upper bounds based on current initial parameters.

            Args:
                initial_parameters: list of floats (initial guess for each pose parameter)

            Return:  
                real_lb, real_ub: floats for lower and upper bound                      
        """

        # Generate real-valued bound constraints for current initial parameters
        self.real_lb = []
        self.real_ub = []

        #for i in range(len(self.to_optimize)):
        #    lb.append(initial_parameters[self.to_optimize[i]] - self.DomainRange[i]) 
        #    ub.append(initial_parameters[self.to_optimize[i]] + self.DomainRange[i])

        #print(' Lower Bound for', self.to_optimize, ' is: ', lb)
        #print(' Upper Bound for', self.to_optimize, ' is: ', ub)

        for i in range(len(self.to_optimize)):
            self.real_lb.append(initial_parameters[self.to_optimize[i]] - self.DomainRange[i]) 
            self.real_ub.append(initial_parameters[self.to_optimize[i]] + self.DomainRange[i])

        for i in range(len(self.to_optimize)):
            print(' \n Bound constraints for ', self.param_sequence[self.to_optimize[i]], ': ', self.real_lb[i], self.real_ub[i])


        ## Define bound constraint for current initial parameters
        #self.opt_LN_NELDERMEAD.set_lower_bounds(lb)
        #self.opt_LN_NELDERMEAD.set_upper_bounds(ub)

        if self.Norm:

            # Assign normalized bound constraints
            self.opt_LN_NELDERMEAD.set_lower_bounds([0.0]*self.Dim)
            self.opt_LN_NELDERMEAD.set_upper_bounds([1.0]*self.Dim)

        else:

            # Assign real bound constraints
            self.opt_LN_NELDERMEAD.set_lower_bounds(self.real_lb)
            self.opt_LN_NELDERMEAD.set_upper_bounds(self.real_ub)        
        
        return self.real_lb, self.real_ub



    def _optimize(self, parameters, verbose = False):

        """Runs the Evolutionary Strategy optimization

            Args:
                parameters: list of floats (initial guess for each pose parameter)            

            Return:      
                minf (float): found minimum
                parameters (list of floats): minimum location                    
        """

        # Pick parameters to optimize only
        parameters_to_optimize = parameters[self.to_optimize]

        print('\n Initial Guess is: ', parameters_to_optimize)

        # Normalize initial parameters if needed
        if self.Norm:
            parameters_to_optimize = list(np.divide(parameters_to_optimize - np.asarray(self.real_lb), (np.asarray(self.real_ub) - np.asarray(self.real_lb))))
    
        # Run optimization
        optimal_parameters = self.opt_LN_NELDERMEAD.optimize(parameters_to_optimize)

        # De-normalized optimal parameters if needed
        if self.Norm:
            optimal_parameters = list(np.multiply((np.asarray(self.real_ub) - np.asarray(self.real_lb)),optimal_parameters) + np.asarray(self.real_lb))

        # Get optimization results
        minf = self.opt_LN_NELDERMEAD.last_optimum_value()

        if verbose:
            print("found minimum at ", optimal_parameters)
            print("minimum value = ", minf)
            print("result code = ", self.opt_LN_NELDERMEAD.last_optimize_result())

        # Update final solution
        parameters[self.to_optimize] = optimal_parameters

        return minf, parameters