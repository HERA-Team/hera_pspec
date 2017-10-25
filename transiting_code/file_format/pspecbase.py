import numpy as np
import parameter as prm

class pspecBase(object):
    def __init__(self):
        """Create properties from pspec attributes."""
        for p in self:
            print p
            this_param = getattr(self, p)
            attr_name = this_param.name
            setattr(self.__class__, attr_name, property(self.prop_fget(p),self.prop_fset(p)))


            
    def __iter__(self):
        """Iterator for all UVParameter attributes."""
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        param_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, prm.psp):
                param_list.append(a)
        for a in param_list:
            yield a
        
    def prop_fget(self, param_name):
        """Getter method for UVParameter properties."""
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.value
        return fget

    def prop_fset(self, param_name):
        """Setter method for UVParameter properties."""
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.value = value
            setattr(self, param_name, this_param)
        return fset