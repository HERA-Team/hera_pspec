import numpy as np

class psparam(object):
    """
    psp == "power spectrum parameter"
    """

    def __init__(self, name, required=True, value=None, form=(), units=None,
                description='', expected_type=None, tols=(1e-5, 1e-08)):
        """Init power spectrum parameter object."""
        self.name = name
        self.required = required
        self.value = value
        self.description = description
        self.units = units
        self.form = form
        if self.form == 'str':
            self.expected_type = str
        else:
            self.expected_type = expected_type
        if np.size(tols) == 1:
            # Only one tolerance given, assume absolute, set relative to zero
            self.tols = (0, tols)
        else:
            self.tols = tols  # relative and absolute tolerances to be used in np.isclose