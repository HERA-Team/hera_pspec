import numpy as np

class UVPSpec(object):
    
    def __init__(self, data_dict=None, attr_dict=None):
        """
        Parameters
        """
        self.data_dict = data_dict
        self.attr_dict = attr_dict
        pass

    def serialize(self):
        """
        Serialize the power spectrum array, supporting data arrays, 
        and relevant properties/attributes from this object into 
        dictionaries.
        
        Returns
        -------
        data_dict : dict
            Dictionary of data arrays for the power spectrum and 
            supporting data.
        
        attr_dict : dict
            Dictionary of attributes and properties of the power 
            spectrum and related data.
        """
        return self.data_dict, self.attr_dict
    
