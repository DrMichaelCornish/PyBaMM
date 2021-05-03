import pybamm
from .ZeroD_Chemistry_1 import ZeroD_Chemistry_1


class HuaEtAl2019(ZeroD_Chemistry_1):
    """
    Zero Dimensional model with Chemistry 1
    
    S_{8}^{0} + 4e^{-} => 2 S_{4}^{2-}
    S_{4}^{2-} + 4e^{-} => S_{2}^{2-} + 2S^{2-}

    along with thermal effects specified in reference [3] below. 
    
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
    
    def __init__(self, options=None, name="Hua et al. (2019)"):
        super().__init__(options, name)
        
        # we change the general model to remove the degradation model
        
        ############### First we need to redefine some local variables and parameters ###############################
        # parameters
        param = self.param
        # standard parameters
        R = param.R
        F = param.F
        T = param.T_ref
        Ta = param.Ta
        N = param.N

        # model-specific known parameters
        Ms = param.Ms
        ns = param.ns
        ns2 = param.ns2
        ns4 = param.ns4
        ns8 = param.ns8
        ne = param.ne
        ih0 = param.ih0
        il0 = param.il0
        rho_s = param.rho_s
        EH0 = param.EH0
        EL0 = param.EL0

        # model-specific unknown parameters
        v = param.v
        ar = param.ar
        k_p = param.k_p
        S_star = param.S_star
        k_s_charge = param.k_s_charge
        k_s_discharge = param.k_s_discharge
        f_s = param.f_s
        c_h = param.c_h
        m_c = param.m_c
        A = param.A
        h = param.h

        i_coef = ne * F / (2 * R * T)
        E_H_coef = R * T / (4 * F)
        f_h = (ns4 ** 2) * Ms * v / ns8
        f_l = (ns ** 2) * ns2 * Ms ** 2 * (v ** 2) / ns4
        
        # variables
        S8 = self.variables["S8 [g]"] 
        S4 = self.variables["S4 [g]"]
        S2 = self.variables["S2 [g]"]
        S = self.variables["S [g]"]
        V = self.variables["Terminal voltage [V]"]
        I = self.variables["Current [A]"]
        Tc = self.variables["Cell Temperature [K]"]
        
        # High plateau potenital [V] as defined by equation (2a) in [1]
        E_H = EH0 + E_H_coef * pybamm.log(f_h * S8 / (S4 ** 2))

        # Low plateau potenital [V] as defined by equation (2b) in [1]
        E_L = EL0 + E_H_coef * pybamm.log(f_l * S4 / (S2 * (S ** 2)))

        # High plateau over-potenital [V] as defined by equation (6a) in [1]
        eta_H = V - E_H

        # Low plateau over-potenital [V] as defined by equation (6b) in [1]
        eta_L = V - E_L

        # High plateau current [A] as defined by equation (5a) in [1]
        i_H = -2 * ih0 * ar * pybamm.sinh(i_coef * eta_H)

        # Low plateau current [A] as defined by equation (5b) in [1]
        i_L = -2 * il0 * ar * pybamm.sinh(i_coef * eta_L)

        # Theoretical capacity [Ah] of the cell as defined by equation (2) in [2]
        cth = (3 * ne * F * S8 / (ns8 * Ms) + ne * F * S4 / (ns4 * Ms)) / 3600

        # Shuttle coefficient (set Tc = T and H = 0 to retrieve Marinescu et al. (2016,2018)
        k_s =   k_s_charge *  (I < 0) * pybamm.exp(-A*N*((1/Tc)-(1/T))/R)
        
        ###################### Next we alter the definition of the dynamic equation ###############################
        
        # New Dynamic Equation for S4
        dS4dt = (ns8 * Ms * i_H / (ne * F)) + k_s * S8 - (ns4 * Ms * i_L / (ne * F))
        self.rhs.update({S4 :dS4dt})
        