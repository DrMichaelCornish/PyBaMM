#
# Marinescu et al (2016) Li-S model
#
import pybamm
from .base_lithium_sulfur_model import BaseModel


class ZeroD_Chemistry_5(BaseModel):
    """
    Zero Dimensional model from Marinescu et al (2016) [1]. Includes S8, S4, S2, S,
    precipitated Li2S (written Sp), and voltage V as direct outputs.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model.
    name : str, optional
        The name of the model.
    """

    def __init__(self, options=None, name="Zero Dimensional Model with Chemistry 5"):
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
        ns8 = param.ns8
        ns6 = 6
        ne = param.ne
        ih0 = param.ih0
        im0 = 5.0
        il0 = param.il0
        rho_s = param.rho_s
        EH0 = param.EH0
        EM0 = param.EM0
        EL0 = param.EL0
        
        m_s = param.S8_initial + param.S6_initial + param.S4_initial + param.S2_initial + param.S_initial + param.Sp_initial

        # model-specific unknown parameters
        v = param.v
        ar = param.ar
        k_p = param.k_p
        S_star = param.S_star
        k_s_charge = param.k_s_charge
        k_s_discharge = param.k_s_discharge
        
        nH = 1
        nM = 1
        nL = 1
        iH_coef = nH * F / (2 * R * Ta)
        iM_coef = nM * F / (2 * R * Ta)
        iL_coef = nL * F / (2 * R * Ta)
        E_H_coef = R * Ta / (nH * F)
        E_M_coef = R * Ta / (nM * F)
        E_L_coef = R * Ta / (nL * F)
        f_h = (ns4 ** (1/6)) * (ns6 ** (1/6)) * (Ms ** (1/3)) * (v ** (1/3)) / (ns8 ** (1/6))
        f_m = (ns2 ** (5/6)) * (Ms ** (1/2)) * (v ** (1/2)) / ((ns4 ** (1/6)) * (ns6 ** (1/6)))
        f_l = ns * (Ms ** (1/2)) * (v ** (1/2)) / (ns2 ** (1/2))

        #######################################################
        # Non-dynamic model functions
        #######################################################

        # High plateau potenital [V] as defined by equation (2a) in [1]
        E_H = EH0 + E_H_coef * pybamm.log(f_h * (S8 ** (1/6)) / ((S6 ** (1/3)) * (S4 ** (1/6))))
        
        E_M = EM0 + E_M_coef * pybamm.log(f_m * (S4 ** (1/6)) * (S6 ** (1/6)) / (S2 ** (5/6)))

        # Low plateau potenital [V] as defined by equation (2b) in [1]
        E_L = EL0 + E_L_coef * pybamm.log(f_l * (S2 ** (1/2)) / S )

        # High plateau over-potenital [V] as defined by equation (6a) in [1]
        eta_H = V - E_H
        
        eta_M = V - E_M

        # Low plateau over-potenital [V] as defined by equation (6b) in [1]
        eta_L = V - E_L
        
        kappa_a = (I>=0)*param.kappa
        kappa_c = (I<0)*param.kappa
        gamma = param.gamma
        
        g8H = 1/(1 + gamma*pybamm.exp(-kappa_a*S8))
        g64H = 1/(1 + gamma*pybamm.exp(-kappa_c*(S6+S4)))
        g64M = 1/(1 + gamma*pybamm.exp(-kappa_a*(S6+S4)))
        g2M = 1/(1 + gamma*pybamm.exp(-kappa_c*S2))
        g2L = 1/(1 + gamma*pybamm.exp(-kappa_a*S2))
        g1L = 1/(1 + gamma*pybamm.exp(-kappa_c*S))
        
        
        i_H = -2*ih0*ar*g8H*g64H* (pybamm.exp(iH_coef*eta_H) - pybamm.exp(-iH_coef*eta_H))
        i_M = -2*im0*ar*g64M*g2M*(pybamm.exp(iM_coef*eta_M) - pybamm.exp(-iM_coef*eta_M))
        i_L = -2*il0*ar*g2L*g1L*(pybamm.exp(iL_coef*eta_L) - pybamm.exp(-iL_coef*eta_L))
        
        
        
        # Theoretical capacity [Ah] of the cell as defined by equation (2) in [2]
        cth = (3 * ne * F * S8 / (ns8 * Ms) + ne * F * S4 / (ns4 * Ms)) / 3600

        # Shuttle coefficient
        k_s = k_s_charge * pybamm.exp(-A*N*((1/Ta)-(1/T))/R) * (I < 0) + k_s_discharge * (I >= 0) * pybamm.exp(-A*N*((1/Ta)-(1/T))/R)

        ###################################
        # Dynamic model functions
        ###################################

        # Algebraic constraint on currents as defined by equation (7) in [1]
        algebraic_condition = i_H + i_M + i_L - I
        self.algebraic.update({V: algebraic_condition})

        # Differential equation (8a) in [1]
        dS8dt = -(ns8 * Ms * i_H / (nH * F))  -k_s*S8
        
        dS6dt = .75*(ns8 * Ms * i_H / (nH * F)) - (ns6 * Ms * i_M / (nM * F)) 

        # Differential equation (8b) in [1]
        dS4dt = .25*(ns8 * Ms * i_H  / (nH * F))  - (ns4 * Ms * i_M / (nM * F)) + k_s*S8

        # Differential equation (8c) in [1]
        dS2dt = 5 * Ms * i_M / (nM * F) - (ns2 * Ms * i_L / (nL * F))

        # Differential equation (8d) in [1]
        dSdt = (ns2 * Ms * i_L / (nL * F)) - k_p * Sp * (S - S_star) / (v * rho_s)

        # Differential equation (8e) in [1]
        dSpdt = k_p * Sp * (S - S_star) / (v * rho_s)

        self.rhs.update({S8 : dS8dt, S6: dS6dt, S4: dS4dt, S2: dS2dt, S: dSdt, Sp: dSpdt})

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
                "Low plateau potential [V]": E_L,
                "High plateau over-potential [V]": eta_H,
                "Low plateau over-potential [V]": eta_L,
                "High plateau current [A]": i_H,
                "Middle plateau current [A]": i_M,
                "Low plateau current [A]": i_L,
                "Theoretical capacity [Ah]": cth,
                "Algebraic condition": algebraic_condition,
                "kappa_a": kappa_a,
                "kappa_c": kappa_c,
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
                self.variables["Terminal voltage [V]"]: param.V_initial,
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
        
        # Add variable for discharge capacity
        """
        Q = pybamm.Variable("Discharge capacity [A.h]")
        self.variables.update({"Discharge capacity [A.h]": Q})
        self.rhs.update({Q: I * self.param.timescale / 3600})
        self.initial_conditions.update({Q: pybamm.Scalar(0)})
        """