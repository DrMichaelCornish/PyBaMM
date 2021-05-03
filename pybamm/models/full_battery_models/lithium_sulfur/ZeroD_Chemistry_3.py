import pybamm
from .base_lithium_sulfur_model import BaseModel


class ZeroD_Chemistry_3(BaseModel):
    """
    Zero Dimensional model with Chemistry 3
    
    3S_{8}^{0} + 8e^{-} => 4 S_{6}^{2-}
    2S_{6}^{2-} + 2e^{-} => 3S_{4}^{2-}
    S_{4}^{2-} + 4e^{-} => S_{2}^{2-} + 2S^{2-}

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
    """

    def __init__(self, options=None, name="Zero Dimensional Model with Chemistry 3"):
        super().__init__(options, name)

        # set external variables
        self.set_external_circuit_submodel()
        V = self.variables["Terminal voltage [V]"]
        I = self.variables["Current [A]"]

        # set internal variables
        S8 = pybamm.Variable("S8 [g]")
        S6 = pybamm.Variable("S6 [g]")
        S4 = pybamm.Variable("S4 [g]")
        S2 = pybamm.Variable("S2 [g]")
        S = pybamm.Variable("S [g]")
        Sp = pybamm.Variable("Precipitated Sulfur [g]")

        #######################################
        # Model parameters as defined in table (1) in [1]. Parameters with 'H' or
        # 'L' in the name represent the high and low plateau parameter, respectively.
        #######################################
        param = self.param
        
        # parameters needed for ks function
        A = param.A
        N = param.N
        Ta = param.Ta
        
        # standard parameters
        R = param.R
        F = param.F
        T = param.T_ref

        # model-specific known parameters
        Ms = param.Ms
        ns = param.ns
        ns2 = param.ns2
        ns4 = param.ns4
        ns6 = 6.0
        ns8 = param.ns8
        ne = 1.0
        ih0 = 10#param.ih0
        il0 = 1#param.il0
        im0 = 5#3*(ih0+il0)/4
        rho_s = param.rho_s
        EH0 = param.EH0
        EL0 = param.EL0
        EM0 = 2.33#3*(EH0+EL0)/4

        # model-specific unknown parameters
        v = param.v
        ar = param.ar
        k_p = param.k_p
        S_star = param.S_star
        k_s_charge = param.k_s_charge
        k_s_discharge = param.k_s_discharge

        iH_coef = ne * F / (2 * R * Ta)
        iL_coef = ne * F / (2 * R * Ta)
        iM_coef = ne * F / (2 * R * Ta)
        E_H_coef = R * Ta / (ne * F)
        E_L_coef = R * Ta / (ne * F)
        E_M_coef = R * Ta / (ne * F)
        f_h = (ns6 ** (4/8)) * (Ms ** (1/8)) * (v ** (1/8)) / (ns8 ** (3/8))
        f_m = (ns4 ** (3/2)) * (Ms ** (1/2)) * (v ** (1/2)) /ns6
        f_l = (ns ** (2/4)) * (ns2 ** (1/4)) * (Ms ** (2/4)) * (v ** (2/4)) / (ns4 ** (1/4)) 
        


        #######################################################
        # Non-dynamic model functions
        #######################################################

        # High plateau potenital [V] as defined by equation (2a) in [1]
        E_H = EH0 + E_H_coef * pybamm.log(f_h * (S8 ** (3/8)) / (S6 ** (4/8)) )
        
        E_M = EM0 + E_M_coef * pybamm.log(f_m * S6 / (S4 ** (3/2)) )

        # Low plateau potenital [V] as defined by equation (2b) in [1]
        E_L = EL0 + E_L_coef * pybamm.log(f_l * (S4 ** (1/4))  / ( (S2 ** (2/4)) * (S ** (1/4))))

        # High plateau over-potenital [V] as defined by equation (6a) in [1]
        eta_H = V - E_H
        
        eta_M = V - E_M

        # Low plateau over-potenital [V] as defined by equation (6b) in [1]
        eta_L = V - E_L

        # High plateau current [A] as defined by equation (5a) in [1]
        i_H = -2 * ih0 * ar * pybamm.sinh(iH_coef * eta_H)
        
        i_M = -2 * im0 * ar * pybamm.sinh(iM_coef * eta_M)

        # Low plateau current [A] as defined by equation (5b) in [1]
        i_L = -2 * il0 * ar * pybamm.sinh(iL_coef * eta_L)

        # Theoretical capacity [Ah] of the cell as defined by equation (2) in [2]
        #cth = (3 * ne * F * S8 / (ns8 * Ms) + ne * F * S4 / (ns4 * Ms)) / 3600

        # Shuttle coefficient
        k_s = k_s_charge * pybamm.exp(-A*N*((1/Ta)-(1/T))/R) * (I < 0) + k_s_discharge * (I >= 0) * pybamm.exp(-A*N*((1/Ta)-(1/T))/R)

        ###################################
        # Dynamic model functions
        ###################################

        # Algebraic constraint on currents as defined by equation (7) in [1]
        algebraic_condition = i_H + i_M + i_L - I
        self.algebraic.update({V: algebraic_condition})

        # Differential equation (8a) in [1]
        dS8dt = -(ns8 * Ms * i_H / (ne * F)) - k_s*S8 

        # Differential equation (8b) in [1]
        dS6dt = (ns8 * Ms * i_H / (ne * F))  - (ns6 * Ms * i_M / (ne * F)) 

        # Differential equation (8b) in [1]
        dS4dt = (ns6 * Ms * i_M / (ne * F))  - (ns4 * Ms * i_L / (ne * F)) + k_s*S8
        
        # Differential equation (8c) in [1]
        dS2dt = ns2 * Ms * i_L / (ne * F)

        # Differential equation (8d) in [1]
        dSdt = 2*(ns * Ms * i_L / (ne * F)) - k_p * Sp * (S - S_star) / (v * rho_s)

        # Differential equation (8e) in [1]
        dSpdt = k_p * Sp * (S - S_star) / (v * rho_s)

        self.rhs.update({S8: dS8dt, S6: dS6dt, S4: dS4dt, S2: dS2dt, S: dSdt, Sp: dSpdt})

        ##############################
        # Model variables
        #############################

        self.variables.update(
            {
                "Time [s]": pybamm.t * self.timescale,
                "S8 [g]": S8,
                "S6 [g]": S6,
                "S4 [g]": S4,
                "S2 [g]": S2,
                "S [g]": S,
                "Precipitated Sulfur [g]": Sp,
                "Shuttle coefficient [s-1]": k_s,
                "Shuttle rate [g-1.s-1]": k_s * S8,
                "High plateau potential [V]": E_H,
                "Mid plateau potential [V]": E_M,
                "Low plateau potential [V]": E_L,
                "High plateau over-potential [V]": eta_H,
                "Mid plateau over-potential [V]": eta_M,
                "Low plateau over-potential [V]": eta_L,
                "High plateau current [A]": i_H,
                "Mid plateau current [A]": i_M,
                "Low plateau current [A]": i_L,
                "Algebraic condition": algebraic_condition,
            }
        )

        ######################################
        # Discharge initial condition
        # The values are found by considering the zero-current
        # state of the battery. Set S8, S4, and Sp as written
        # below. Then, solve eta_H = V, eta_L = V, the algebraic
        # condition, and mass conservation for the remaining values.
        ######################################

        self.initial_conditions.update(
            {
                self.variables["S8 [g]"]: param.S8_initial,
                self.variables["S6 [g]"]: param.S6_initial,
                self.variables["S4 [g]"]: param.S4_initial,
                self.variables["S2 [g]"]: param.S2_initial,
                self.variables["S [g]"]: param.S_initial,
                self.variables["Precipitated Sulfur [g]"]: param.Sp_initial,
                self.variables["Terminal voltage [V]"]: param.V_initial
            }
        )

        ######################################
        # Model events
        ######################################
        
        tol = 1e-4
        self.events.append(
            pybamm.Event(
                "Minimum voltage",
                V - self.param.voltage_low_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        
        self.events.append(
            pybamm.Event(
                "Maximum voltage",
                V - self.param.voltage_high_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        
        #self.events.append(
        #    pybamm.Event(
        #        "Zero theoretical capacity", cth - tol, pybamm.EventType.TERMINATION
        #    )
        #)
