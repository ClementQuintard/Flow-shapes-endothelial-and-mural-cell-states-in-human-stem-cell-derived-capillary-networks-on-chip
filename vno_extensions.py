"""Extensions to vno_model.py: (2) Michaelis-Menten / transfer-limited O2 along the channel,
(5) loop re-oxygenation with liquid-side resistance, (6) sensitivity sweep.
Run from the package directory:  python vno_extensions.py
Imports PARAMS from vno_model.py so every number shares one source."""
import math, copy, sys, numpy as np, pandas as pd
from scipy.optimize import brentq
from vno_model import PARAMS, BARRER, ul, channel

P = PARAMS; Csat = P['Csat']; beta = P['beta']
THRESH = 0.10   # a trap is 'supplied' if its tissue-surface O2 >= 0.10 Csat (~14 mmHg). Fixed before any result.
NSUB = 20       # plug-flow sub-steps per trap

# ---------------- (2) O2 along the channel: per-trap balance, series resistances ----------------
def q_eff(Cw, qmax, Km=1e-3, alpha=0.0, Ch=20e-3, nH=2):
    """Michaelis-Menten (Km, mol/m3) times hypoxic down-regulation H = (1-alpha) + alpha*Cw^n/(Ch^n+Cw^n)."""
    M = Cw/(Km+Cw) if Km > 0 else 1.0
    H = (1-alpha) + alpha*Cw**nH/(Ch**nH + Cw**nH)
    return qmax*M*H

def J_tissue(Cw, p, qmax, Aface=None, transfer_limited=True, **kin):
    """Uptake of one trap's tissue at surface concentration Cw: min(kinetic demand, zero-order penetration-limited flux).
    Slab of area Aface (default Aint) and depth LT; penetration depth delta = sqrt(2 D Cw/(nc q))."""
    Nt = p['N']/p['ntrap']; qe = q_eff(Cw, qmax, **kin); Jkin = Nt*qe
    if not transfer_limited: return Jkin
    if qe <= 0 or Cw <= 0: return 0.0
    A = p['Aint'] if Aface is None else Aface
    delta = math.sqrt(2*p['D']*Cw/(p['nc']*qe))
    return min(Jkin, A*p['nc']*qe*min(delta, p['LT']))

def trap_profile(p, Qul, qmax, Cin, transfer_limited=True, Aface=None, Sh=None, **kin):
    """Plug-flow march along the 10 traps. Wall transfer k = max(Sh*D/h, Leveque). Returns per-trap arrays."""
    Q = ul(Qul); c = channel(p, Qul); A = p['Aint'] if Aface is None else Aface
    k_lev = 0.807*(c['gam']*p['D']**2/p['Ltrap'])**(1/3)
    k = max((p['Sh_LF'] if Sh is None else Sh)*p['D']/p['h'], k_lev); kA = k*A
    def flux(Cb):
        if Cb <= 0: return 0.0, 0.0
        if not transfer_limited: return J_tissue(Cb, p, qmax, A, False, **kin), Cb
        g = lambda Cw: kA*(Cb-Cw) - J_tissue(Cw, p, qmax, A, True, **kin)
        Cw = brentq(g, 0.0, Cb) if g(0.0) > 0 else 0.0
        return kA*(Cb-Cw), Cw
    Cb_in, Cb_out, Cw_l, J_l, x_l, C_l = [], [], [], [], [], []
    C = Cin
    for n in range(p['ntrap']):
        Cb_in.append(C); Jsum = 0.0; Cw_first = None
        for s in range(NSUB):
            J, Cw = flux(C)
            if Cw_first is None: Cw_first = Cw
            dC = J/NSUB/Q; Jsum += J/NSUB if dC <= C else C*Q
            C = max(0.0, C - dC); x_l.append(n + (s+1)/NSUB); C_l.append(C)
        Cb_out.append(C); Cw_l.append(Cw_first); J_l.append(Jsum)
    return dict(Cb_in=np.array(Cb_in), Cb_out=np.array(Cb_out), Cw=np.array(Cw_l), J=np.array(J_l), k=k, x=np.array(x_l), C=np.array(C_l))

def summarize(r, p, qmax):
    return int(np.sum(r['Cw'] >= THRESH*Csat)), r['J'].sum()/(p['N']*qmax)

VARIANTS = {
 'A zero-order, no wall limit (note)':          dict(transfer_limited=False, Km=0),
 'B Michaelis-Menten Km 1 uM, no wall limit':   dict(transfer_limited=False, Km=1e-3),
 'C wall transfer + penetration limited, MM':   dict(transfer_limited=True, Km=1e-3),
 'D C + hypoxic down-regulation alpha 0.5':     dict(transfer_limited=True, Km=1e-3, alpha=0.5),
 'E C + hypoxic down-regulation alpha 0.9':     dict(transfer_limited=True, Km=1e-3, alpha=0.9),
}

# ---------------- (5) Loop re-oxygenation with liquid-side (Graetz) resistance ----------------
def Q_wall(t, Pbarrer): return 2*math.pi*t['L']*Pbarrer*BARRER/(beta*math.log(t['ro']/t['ri']))
def Sh_hausen(Qul, t):
    d = 2*t['ri']; U = ul(Qul)/(math.pi*t['ri']**2); Gz = U*d/P['D']*d/t['L']
    return 3.66 + 0.0668*Gz/(1+0.04*Gz**(2/3))
def Q_liq(Qul, t): return Sh_hausen(Qul, t)*math.pi*P['D']*t['L']       # = Sh*pi*D*L, independent of diameter
def Q_ex_series(Qul, t, Pbarrer, liquid=True):
    qw = Q_wall(t, Pbarrer); return 1/(1/qw + 1/Q_liq(Qul, t)) if liquid else qw
def inlet_frac(Qul, q, tubes, kL, liquid=True):
    """tubes = [(tube dict with L, ri, ro; P in Barrer), ...]. Returns (C_in/C_sat, total Q_ex m3/s)."""
    Q = ul(Qul); Qex_tot = sum(Q_ex_series(Qul, t, Pb, liquid) for t, Pb in tubes)
    f = math.exp(-Qex_tot/Q)*Q/(Q + kL*P['Ares'])
    return 1 - f/(1-f)*P['N']*q/Q/Csat, Qex_tot
SIL = dict(ri=0.38e-3, ro=0.825e-3)   # 0.76 mm ID x 1.65 mm OD silicone
def inlet_with_sil(Qul, q, L, geom=SIL, Psil=600, liquid=True):
    return inlet_frac(Qul, q, [(P['Tygon'],16), (P['PTFE'],4), (dict(L=L, **geom), Psil)], 2e-6, liquid)[0]

# ---------------- (6) Sensitivity ----------------
def evaluate(p, q=None, kL=(2e-6,10e-6), PTFE_P=(4,45), Sh=None, Aface=None, LBscale=1.0, model_kw=dict(transfer_limited=True, Km=1e-3)):
    q = p['q']['central'] if q is None else q; res = {}
    r = trap_profile(p, 1.0, q, Csat, Sh=Sh, Aface=Aface, **model_kw)
    res['LF traps supplied (of 10)'], res['LF demand met'] = summarize(r, p, q); res['LF outlet C/Csat'] = r['Cb_out'][-1]/Csat
    lo = inlet_frac(1000.0, q, [(p['Tygon'],16),(p['PTFE'],PTFE_P[0])], kL[0])[0]; res['HF inlet C/Csat (worst loop)'] = lo
    r = trap_profile(p, 1000.0, q, lo*Csat, Sh=Sh, Aface=Aface, **model_kw)
    res['HF traps supplied (of 10)'], res['HF demand met by face diffusion'] = summarize(r, p, q); res['HF C_wall/Csat trap 1'] = r['Cw'][0]/Csat
    c = channel(p, 1000.0)
    for tr, dP in (('1', c['dP1']), ('10', c['dP10'])):
        u = p['kappa']*dP*LBscale/(p['mu']*p['LT'])
        res[f'HF interstitial O2 / trap demand, trap {tr}'] = u*p['wT']*p['h']*Csat/(p['N']/p['ntrap']*q)
    res['HF capillary tau trap 1 (dyn/cm2)'] = c['dP1']*LBscale*p['d_cap']/(4*p['Lv'])*10
    res['HF Darcy u trap 1 (um/s)'] = p['kappa']*c['dP1']*LBscale/(p['mu']*p['LT'])*1e6
    return res
SWEEPS = {
 'kappa 1e-15 / 1e-13 m2': [dict(p_over=dict(kappa=1e-15)), dict(p_over=dict(kappa=1e-13))],
 'q 2e-17 / 1e-16 mol/s/cell': [dict(q=2e-17), dict(q=1e-16)],
 'N 1e5 / 1e6 cells': [dict(p_over=dict(N=1e5, nc=P['nc']/5)), dict(p_over=dict(N=1e6, nc=P['nc']*2))],
 'L_B(n) x0.7 / x1.3': [dict(LBscale=0.7), dict(LBscale=1.3)],
 'k_L 1 / 20 um/s': [dict(kL=(1e-6,1e-6)), dict(kL=(20e-6,20e-6))],
 'PTFE 4 / 45 Barrer (worst loop)': [dict(PTFE_P=(4,4)), dict(PTFE_P=(45,45))],
 'Sh_LF 2 / 6': [dict(Sh=2.0), dict(Sh=6.0)],
 'interface area 0.4 / 0.5 mm2': [dict(Aface=4e-7), dict(Aface=5e-7)],
 'hypoxic down-reg alpha 0 / 0.9': [dict(model_kw=dict(transfer_limited=True, Km=1e-3, alpha=0.0)), dict(model_kw=dict(transfer_limited=True, Km=1e-3, alpha=0.9))],
}
def sensitivity():
    base = evaluate(P); rows = []
    for name, (lo, hi) in SWEEPS.items():
        vals = []
        for side in (lo, hi):
            p2 = copy.deepcopy(P); p2.update(side.get('p_over', {}))
            vals.append(evaluate(p2, **{k:v for k,v in side.items() if k != 'p_over'}))
        for out in base: rows.append(dict(parameter=name, output=out, baseline=base[out], low=vals[0][out], high=vals[1][out]))
    return base, pd.DataFrame(rows)

if __name__ == '__main__':
    qc, qh = P['q']['central'], P['q']['high']
    print('== (2) O2 along the channel, central uptake ==')
    for lab, kw in VARIANTS.items():
        for Qul, cin in ((1.0, Csat), (1000.0, 0.951*Csat)):
            r = trap_profile(P, Qul, qc, cin, **kw); s, dm = summarize(r, P, qc)
            print(f'{lab:46s} Q={Qul:5g} Cw/Csat {np.round(r["Cw"]/Csat,2)} outlet {r["Cb_out"][-1]/Csat:.2f} supplied {s}/10 demand met {dm:.2f}')
    print('LF per-trap transfer fraction kA/Q = %.3f' % (trap_profile(P,1.0,qc,Csat)['k']*P['Aint']/ul(1)))
    print('\n== (5) Loop ==')
    Ty, Pt = P['Tygon'], P['PTFE']
    for liq in (False, True):
        print('liquid side %5s: Q_ex Tygon %.0f, PTFE(4) %.0f, PTFE(45) %.0f ul/min' % (liq, Q_ex_series(1000,Ty,16,liq)*6e10, Q_ex_series(1000,Pt,4,liq)*6e10, Q_ex_series(1000,Pt,45,liq)*6e10))
    for ql, q in P['q'].items():
        print(f'  HF inlet {ql:7s}: {inlet_frac(1000,q,[(Ty,16),(Pt,4)],2e-6)[0]:.3f}-{inlet_frac(1000,q,[(Ty,16),(Pt,45)],10e-6)[0]:.3f}')
    for Ps in (300, 600, 800):
        L98 = brentq(lambda L: inlet_with_sil(1000, qh, L, SIL, Ps) - 0.98, 1e-3, 50); L98w = brentq(lambda L: inlet_with_sil(1000, qh, L, SIL, Ps, False) - 0.98, 1e-3, 50)
        print(f'  silicone 0.76 mm ID, {Ps} Barrer: {L98*100:.1f} cm for worst-case inlet 0.98 (wall-only estimate {L98w*100:.1f} cm)')
    print('  silicone pump tubing (same dims) alone: %.3f' % inlet_frac(1000, qh, [(dict(L=Ty['L'],ri=Ty['ri'],ro=Ty['ro']),600),(Pt,4)], 2e-6)[0])
    print('\n== (6) Sensitivity ==')
    base, sens = sensitivity(); sens['span'] = (sens.high-sens.low).abs()
    for out in base:
        top = sens[(sens.output==out) & (sens.span > 1e-9)].sort_values('span', ascending=False)
        print(f'{out}: baseline {base[out]:.3g}; moved by: ' + ', '.join(f'{r.parameter} ({r.low:.3g}->{r.high:.3g})' for _, r in top.iterrows()) if len(top) else f'{out}: baseline {base[out]:.3g}; robust to all sweeps')
