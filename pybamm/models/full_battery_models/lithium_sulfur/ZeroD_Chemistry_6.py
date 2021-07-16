import pybamm
from .base_lithium_sulfur_model import BaseModel


class ZeroD_Chemistry_6(BaseModel):
    """
    Zero Dimensional model from Zhang et al (2015) [1].
    
    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model.
    name : str, optional
        The name of the model.

    References
    ----------
    [1] Zhang, T., Marinescu, M., O’Neill, L., Wild, M., & Offer, G. (2015). 
        Modeling the voltage loss mechanisms in lithium-sulfur cells: The importance 
        of electrolyte resistance and precipitation kinetics. Physical Chemistry 
        Chemical Physics, 17(35), 22581–22586. https://doi.org/10.1039/c5cp03566j
    
    """

    def __init__(self, options=None, name="Zhang et al. (2015) model"):
        super().__init__(options, name)

        # set external variables
        self.set_external_circuit_submodel()
        #V = self.variables["Terminal voltage [V]"]
        I = self.variables["Current [A]"]

        # set internal variables
        S8s = pybamm.Variable("S8(s) [g]")
        S8 = pybamm.Variable("S8 [g]")
        S6 = pybamm.Variable("S6 [g]")
        S4 = pybamm.Variable("S4 [g]")
        S2 = pybamm.Variable("S2 [g]")
        S = pybamm.Variable("S [g]")
        eta2 = pybamm.Variable("Reaction 2 overpotential [V]") 
        

        #######################################
        # Model parameters as defined in table (1) in [1]. Parameters with 'H' or
        # 'L' in the name represent the high and low plateau parameter, respectively.
        #######################################
        param = self.param

        # standard parameters
        R = param.R
        F = param.F
        T = param.T_ref
        N = param.N

        # model-specific known parameters
        '''
        For the stochiometric sij paramters, i indicates species of Sulfur and j reaction number.
        For example, S6 in reaction 3 has s63.
        The one exception is for S8(s), which has S instead of an integer. 
        '''
        sLi1 = param.sLi1
        sS2 = param.sS2
        s82 = param.s82
        s83 = param.s83
        s63 = param.s63
        s64 = param.s64
        s44 = param.s44
        s45 = param.s45
        s25 = param.s25
        s26 = param.s26
        s16 = param.s16
        
        E01 = param.E01
        E02 = param.E02
        E03 = param.E03
        E04 = param.E04
        E05 = param.E05
        E06 = param.E06
        
        n1 = param.n1
        n2 = param.n2
        n3 = param.n3
        n4 = param.n4
        n5 = param.n5
        n6 = param.n6
        
        i02 = param.i02
        i03 = param.i03
        i04 = param.i04
        i05 = param.i05
        i06 = param.i06
        
        A = param.Area
        l = param.l
        V_Li2S = param.V_Li2S
        nu0_Li2S = param.nu0_Li2S
        Li0 = param.Li0
        k_p = param.k_p
        K_sp = param.K_sp
        av0 = param.av0
        sigma0 = param.sigma0
        b = param.b
        eta1 = param.eta1
        
        
        #######################################################
        # Non-dynamic model functions
        #######################################################
        
        # Lithium ion concentration as defined in equation (10) in [1]
        Li =  2*(S8 + S6 + S4 + S2 + S) + Li0 
        
        # Reduction Potential as given in Equation (14) of [1]
        E1 = E01 - (R*T/(n1*F))*sLi1*pybamm.log(Li/1000) 
        
        # Reduction Potential as given in Equation (14) of [1]
        E2 = E02 - (R*T/(n2*F))*( sS2*pybamm.log(S8s/1000) + s82*pybamm.log(S8/1000) )
        
        # Reduction Potential as given in Equation (14) of [1]
        E3 = E03 - (R*T/(n3*F))*( s83*pybamm.log(S8/1000) + s63*pybamm.log(S6/1000) )
        
        # Reduction Potential as given in Equation (14) of [1]
        E4 = E04 - (R*T/(n4*F))*( s64*pybamm.log(S6/1000) + s44*pybamm.log(S4/1000) )
        
        # Reduction Potential as given in Equation (14) of [1]
        E5 = E05 - (R*T/(n5*F))*( s45*pybamm.log(S4/1000) + s25*pybamm.log(S2/1000) )
        
        # Reduction Potential as given in Equation (14) of [1]
        E6 = E06 - (R*T/(n6*F))*( s26*pybamm.log(S2/1000) + s16*pybamm.log(S/1000) )
        
        # Volume fraction function of Li2S defined in equation (11) in [1]
        nu_Li2S = nu0_Li2S #- ep + ep0
        
        # Precipitation rate defined in equation (11) in [1]
        r_p = k_p*nu_Li2S*((Li**2)*S - K_sp)
        
        # Active surface area as defined in equation (16) in [1]
        av = av0#*((ep/ep0)**xi)
        
        # Electrolyte conductivity as defined in equation (15) in [1]
        sigma = (sigma0 - b*pybamm.AbsoluteValue(Li-Li0)) #* (ep**(1.5))  
        
        # Electrolyte Resistance is given in [1] on page 3
        Rs = l/(A*sigma)
        
        # Voltage as defined by equation (8) in [1]
        V = E2 + eta2 -(E1 + eta1) - I*Rs
        #eta2 = V - ( E2 -(E1 + eta1) - I*Rs ) 
        
        # Overpotential implied by equation (8) in [1]
        eta3 = E2 + eta2 - E3
        
        # Overpotential implied by equation (8) in [1]
        eta4 = E2 + eta2 - E4
        
        # Overpotential implied by equation (8) in [1]
        eta5 = E2 + eta2 - E5
        
        # Overpotential implied by equation (8) in [1]
        eta6 = E2 + eta2 - E6
        
        # Reaction current functions defined in equation (12) in [1]
        i2 = 2*i02*pybamm.sinh(n2*F*eta2/(2*R*T))
        
        # Reaction current functions defined in equation (12) in [1]
        i3 = 2*i03*pybamm.sinh(n3*F*eta3/(2*R*T))
        
        # Reaction current functions defined in equation (12) in [1]
        i4 = 2*i04*pybamm.sinh(n4*F*eta4/(2*R*T))
        
        # Reaction current functions defined in equation (12) in [1]
        i5 = 2*i05*pybamm.sinh(n5*F*eta5/(2*R*T))
        
        # Reaction current functions defined in equation (12) in [1]
        i6 = 2*i06*pybamm.sinh(n6*F*eta6/(2*R*T))
        
        ###################################
        # Dynamic model functions
        ###################################

        # Algebraic constraint on currents as defined by equation (13) in [1]
        algebraic_condition = av*A*l*(i2 + i3 + i4 + i5 + i6) - I
        self.algebraic.update({eta2: algebraic_condition})

        # Differential equation (9) in [1]
        dS8sdt = (av/F)*(sS2*i2/n2)

        # Differential equation (9) in [1]
        dS8dt = (av/F)*( (s82*i2/n2) + (s83*i3/n3) )

        # Differential equation (9) in [1]
        dS6dt = (av/F)*( (s63*i3/n3) + (s64*i4/n4) )
        
        # Differential equation (9) in [1]
        dS4dt = (av/F)*( (s44*i4/n4) + (s45*i5/n5) )

        # Differential equation (9) in [1]
        dS2dt = (av/F)*( (s25*i5/n5) + (s26*i6/n6) )

        # Differential equation (9) in [1]
        dSdt = (av/F)*(s16*i6/n6) - r_p
        
        self.rhs.update({S8s: dS8sdt,
                         S8: dS8dt,
                         S6 : dS6dt,
                         S4: dS4dt, 
                         S2: dS2dt, 
                         S: dSdt
                        })

        ##############################
        # Model variables
        #############################

        self.variables.update(
            {
                "Time [s]": pybamm.t * self.timescale,
                "S8(s) [g]": S8s,
                "S8 [g]": S8,
                "S6 [g]": S6,
                "S4 [g]": S4,
                "S2 [g]": S2,
                "S [g]": S,
                "Li [g]" : Li,
                "Terminal voltage [V]" : V,
                "Electrolyte Resistance [S-1]" : Rs,
                "Conductivity [m.S-1]" : sigma,
                "Active Surface Area [m2]" : av,
                "Reaction 2 potential [V]": E2,
                "Reaction 3 potential [V]": E3,
                "Reaction 4 potential [V]": E4,
                "Reaction 5 potential [V]": E5,
                "Reaction 6 potential [V]": E6,
                "Reaction 1 overpotential [V]": eta1,
                "Reaction 2 overpotential [V]": eta2,
                "Reaction 3 overpotential [V]": eta3,
                "Reaction 4 overpotential [V]": eta4,
                "Reaction 5 overpotential [V]": eta5,
                "Reaction 6 overpotential [V]": eta6,
                "Reaction 2 current [A]": i2,
                "Reaction 3 current [A]": i3,
                "Reaction 4 current [A]": i4,
                "Reaction 5 current [A]": i5,
                "Reaction 6 current [A]": i6,
                "Algebraic condition": algebraic_condition
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
                self.variables["S8(s) [g]"]: param.zhang_S8s_initial,
                self.variables["S8 [g]"]: param.zhang_S8_initial,
                self.variables["S6 [g]"]: param.zhang_S6_initial,
                self.variables["S4 [g]"]: param.zhang_S4_initial,
                self.variables["S2 [g]"]: param.zhang_S2_initial,
                self.variables["S [g]"]: param.zhang_S_initial,
                self.variables["Reaction 2 overpotential [V]"]: param.zhang_eta2_initial          
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