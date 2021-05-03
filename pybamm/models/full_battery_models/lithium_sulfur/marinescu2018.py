import pybamm
from .ZeroD_Chemistry_1 import ZeroD_Chemistry_1


class MarinescuEtAl2018(ZeroD_Chemistry_1):
    """
    Zero Dimensional model with Chemistry 1
    
    S_{8}^{0} + 4e^{-} => 2 S_{4}^{2-}
    S_{4}^{2-} + 4e^{-} => S_{2}^{2-} + 2S^{2-}

    along with degradation effects specified in reference [2] below. 
    
    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model.
    name : str, optional
        The name of the model.

    References
    ----------
    .. [1]  Marinescu, M., Zhang, T. & Offer, G. J. (2016).
            A zero dimensional model of lithium-sulfur batteries during charge
            and discharge. Physical Chemistry Chemical Physics, 18, 584-593.

    .. [2]  Marinescu, M., O’Neill, L., Zhang, T., Walus, S., Wilson, T. E., &
            Offer, G. J. (2018).
            Irreversible vs reversible capacity fade of lithium-sulfur batteries
            during cycling: the effects of precipitation and shuttle. Journal of
            The Electrochemical Society, 165(1), A6107-A6118.
            
    .. [3]  Hua, X., Zhang, T., Offer, G., & Marinescu, M. (2019).
            Towards online tracking of the shuttle effect in lithium sulfur 
            batteries using differential thermal voltemetry. Journal of Energy 
            Storage, 21 (2019), 765-772.
    """
    
    def __init__(self, options=None, name="Marinescu et al. (2018)"):
        super().__init__(options, name)
        
        '''
        This model is effectively chemistry 1 without the Hua et al. (2019) temperature features
        and without the Marinescu et al. (2018) degredation features. 
        Therefore, we copy the Marinescu et al. (2018) code to remove temperature and copy
        the Hua et al. (2019) code to remove degredation. 
        '''
        
        
        # we change the general model to have no temperature dynamics. 
        param = self.param
        Tc = self.variables['Cell Temperature [K]']
        dTcdt = 0*Tc
        self.rhs.update({Tc :dTcdt})
        self.initial_conditions.update(
            {
                self.variables["Cell Temperature [K]"] : param.Ta
            }
        )