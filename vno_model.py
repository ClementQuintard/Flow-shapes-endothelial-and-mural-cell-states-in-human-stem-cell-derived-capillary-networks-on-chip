"""Analytical flow and O2 transport model for the serpentine trap chip (LF vs HF).
Single source of truth for every number in the Flow and O2 transport note.
Run: python vno_model.py   (prints all tables; edit PARAMS to explore)."""
import math

PARAMS = dict(
    Q_LF=1.0, Q_HF=1000.0,          # flow rates (ul/min)
    h=500e-6, L=51e-3,              # square channel side, channel length (m)
    LB1=40e-3, LB10=13e-3,          # bypass length across trap 1 / trap 10 (m) [to confirm]
    ntrap=10, N=5e5,                # traps per channel, cells per channel (measured)
    q=dict(low=2e-17, central=4e-17, high=1e-16),  # O2 sink per cell (mol/s)
    Csat=0.18, beta=1.3e-3/133.32,  # O2 saturation (mol/m3), solubility (mol m-3 Pa-1)
    D=2e-9, mu=0.8e-3, rho=1000.0,  # O2 diffusivity, viscosity, density (SI)
    PoB=14.227,                     # Poiseuille number, square duct (Shah & London)
    kappa=1e-14, wT=800e-6, LT=700e-6,  # ECM permeability, ECM-filled trap width/length (m)
    Aint=0.8e-3*0.5e-3, Ltrap=1e-3,      # trap/medium interface area, trap length (m2, m)
    d_cap=10e-6, d_art=30e-6, Lv=800e-6, # vessel diameters and path length (m)
    nc=5e4/(0.8*0.8*0.5)*1e9,            # sink density in trap (cells/m3)
    Tygon=dict(L=0.406, ri=0.51e-3, ro=1.36e-3, P=16),       # pump tubing, P in Barrer
    PTFE=dict(L=0.60, ri=0.395e-3, ro=0.795e-3, P=(4, 45)),  # return tubing
    kL=(2e-6, 10e-6), Ares=58e-6,                             # reservoir mass transfer
    Sh_LF=3.15,                                               # fully developed Sherwood (LF)
)
BARRER = 3.35e-16  # mol m / (m2 s Pa)
ul = lambda x: x * 1e-9 / 60

def channel(p, Qul):
    Q = ul(Qul); h = p['h']; U = Q / h**2
    G = 2 * p['PoB'] * p['mu'] * Q / h**4            # pressure gradient (Pa/m)
    tau = G * h / 4                                    # perimeter-mean wall shear (Pa)
    return dict(U=U, Re=p['rho']*U*h/p['mu'], Pe=U*h/p['D'], tres=p['L']/U, tdiff=h**2/p['D'],
                G=G, tau=tau, gam=tau/p['mu'], tau_mid=1.347*tau,
                dP1=G*p['LB1'], dP10=G*p['LB10'])

def Qex(p, t, P):
    return 2*math.pi*t['L']*P*BARRER/(p['beta']*math.log(t['ro']/t['ri']))*6e10   # ul/min

def inlet(p, Qul, q, Qp, kla):
    f = math.exp(-Qex(p, p['Tygon'], p['Tygon']['P'])/Qul)*math.exp(-Qp/Qul)*Qul/(Qul+kla)
    return 1 - f/(1-f)*p['N']*q/ul(Qul)/p['Csat']

def run(p=PARAMS):
    print('Return path: Q_ex Tygon %.0f ul/min, PTFE %.0f-%.0f, kLA %.0f-%.0f' % (
        Qex(p, p['Tygon'], 16), Qex(p, p['PTFE'], 4), Qex(p, p['PTFE'], 45), p['kL'][0]*p['Ares']*6e10, p['kL'][1]*p['Ares']*6e10))
    Qp = (Qex(p, p['PTFE'], 4), Qex(p, p['PTFE'], 45)); kla = (p['kL'][0]*p['Ares']*6e10, p['kL'][1]*p['Ares']*6e10)
    for lab, Qul in (('LF', p['Q_LF']), ('HF', p['Q_HF'])):
        c = channel(p, Qul); Q = ul(Qul)
        print(f'\n== {lab}: Q = {Qul:g} ul/min ==')
        print(f"U {c['U']*1e3:.3g} mm/s | Re {c['Re']:.2g} | Pe {c['Pe']:.3g} | t_res {c['tres']:.3g} s | t_res/t_diff {c['tres']/c['tdiff']:.2g}")
        print(f"wall shear: mean {c['tau']*10:.3g} dyn/cm2 (mid-wall {c['tau_mid']*10:.3g}), shear rate {c['gam']:.3g} 1/s")
        print(f"dP across trap 1 / 10: {c['dP1']:.3g} / {c['dP10']:.3g} Pa")
        for d, nm in ((p['d_art'], 'arteriole 30um'), (p['d_cap'], 'capillary 10um')):
            t1, t10 = c['dP1']*d/(4*p['Lv']), c['dP10']*d/(4*p['Lv'])
            v1 = t1*d/(8*p['mu'])
            print(f"  {nm}: tau {t1*10:.3g} -> {t10*10:.3g} dyn/cm2, gamma {t1/p['mu']:.3g} -> {t10/p['mu']:.3g} 1/s, Re {p['rho']*v1*d/p['mu']:.2g} (trap1)")
        for tr, dP in (('1', c['dP1']), ('10', c['dP10'])):
            u = p['kappa']*dP/(p['mu']*p['LT'])
            print(f"  Darcy trap {tr}: u {u*1e6:.3g} um/s, Q_T {u*p['wT']*p['h']*6e10:.3g} ul/min, Pe_int {u*250e-6/p['D']:.2g}, tau_int {math.sqrt(p['kappa'])*dP/p['LT']*10:.2g} dyn/cm2, O2 {u*p['wT']*p['h']*p['Csat']:.2g} mol/s")
        k = p['Sh_LF']*p['D']/p['h']*1.0 if lab == 'LF' else 0.807*(c['gam']*p['D']**2/p['Ltrap'])**(1/3)
        Jmax = k*p['Csat']*p['Aint']
        print(f"Leveque k {k*1e6:.3g} um/s, J_max/trap {Jmax:.2g} mol/s")
        for nm, q in p['q'].items():
            Da = p['N']*q/(Q*p['Csat']); Jt = p['N']/p['ntrap']*q
            cin = [inlet(p, Qul, q, Qp[i], kla[i]) for i in (0, 1)]
            delta = math.sqrt(2*p['D']*p['Csat']/(p['nc']*q))
            print(f"  {nm:7s} Da {Da:.3g} | traps reached {min(10, 10/Da):.2g} | outlet C/Cin {max(0, 1-Da):.3g} | inlet C/Csat {min(cin):.3f}-{max(cin):.3f} | C_wall/C_bulk {1-Jt/Jmax:.2f} | delta {delta*1e6:.0f} um")

if __name__ == '__main__':
    run()
