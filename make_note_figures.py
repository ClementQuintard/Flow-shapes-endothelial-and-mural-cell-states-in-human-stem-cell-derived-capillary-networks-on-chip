"""Figures 1-3 and the raw-data workbook of the Supplementary Note 'Flow and O2 transport in VNOs-on-chip' (v3).
Figure 1  fig_Da.png                      Damkohler number vs flow rate (section 4)
Figure 2  fig_ext1_O2_profile_models.png  O2 along the channel, transport-limited model vs bound (section 5)
Figure 3  fig_ext3_sensitivity_tornado.png sensitivity of three conclusions (section 7)
          VNO_note_v3_raw_data.xlsx        one sheet per panel
Run from the physics package directory (needs vno_model.py and vno_extensions.py): python make_note_figures.py
All numbers come from PARAMS in vno_model.py and the functions of vno_extensions.py; inlet O2 = C_sat."""
import math, copy, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import vno_extensions as E
from vno_model import channel, ul
P, Csat = E.P, E.Csat; qc = P['q']['central']; Lmm = P['L']*1e3
LF, HF, G1, G2, G3 = '#1f4e9c', '#00b8d9', '#9a9a9a', '#5a5a5a', '#c8c8c8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8})
def letter(ax, s): ax.text(-0.12, 1.05, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')
xl = {}

# ---------------- Figure 1: Da vs Q ----------------
Qs = np.logspace(math.log10(0.5), math.log10(3000), 200)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
d1 = pd.DataFrame({'Q_ul_per_min': Qs})
for lab, q, ls in (('low uptake', P['q']['low'], '--'), ('central uptake', qc, '-'), ('high uptake', P['q']['high'], ':')):
    da = P['N']*q/(ul(Qs)*Csat); d1[f'Da {lab}'] = da
    ax.plot(Qs, da, color=G2, ls=ls, lw=1.6, label=lab)
ax.axhline(1, color=G3, lw=1); ax.text(0.55, 1.15, 'Da = 1', color=G2, fontsize=7)
ax.axvline(P['Q_LF'], color=LF, lw=1.5); ax.text(1.1, 40, 'LF\n1 µl/min', color=LF, fontsize=7, va='top')
ax.axvline(P['Q_HF'], color=HF, lw=1.5); ax.text(1100, 0.3, 'HF\n1,000 µl/min', color=HF, fontsize=7, va='top')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Flow rate Q (µl/min)'); ax.set_ylabel('Damköhler number Da')
ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(['1', '10', '100', '1000']); ax.set_yticks([0.001, 0.01, 0.1, 1, 10]); ax.set_yticklabels(['0.001', '0.01', '0.1', '1', '10'])
ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.80, 1.0))
fig.savefig('fig_Da.png', dpi=300, bbox_inches='tight'); plt.close(fig); xl['Fig1_Da_vs_Q'] = d1

# ---------------- Figure 2: O2 along the channel ----------------
models = [('No transport limit (bound)', dict(transfer_limited=False, Km=0)), ('Transport-limited model', dict(transfer_limited=True, Km=1e-3))]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.5))
for ax, (lab, Qul, col) in zip(axes[:2], [('LF', P['Q_LF'], LF), ('HF', P['Q_HF'], HF)]):
    df = None
    for vname, kw in models:
        r = E.trap_profile(P, Qul, qc, Csat, **kw); x = r['x']/P['ntrap']*Lmm; bound = not kw['transfer_limited']
        ax.plot(x, r['C']/Csat, ls=':' if bound else '-', color=G2 if bound else col, lw=1.1 if bound else 1.6, label=f'{vname}: medium')
        if not bound:
            ax.plot((np.arange(10)+0.05)/10*Lmm, r['Cw']/Csat, 'o', color=col, ms=3.5, mfc='white', mew=1.2, label='Transport-limited model: tissue surface at each trap')
            xl[f'Fig2_{lab}_tissue_surface'] = pd.DataFrame({'trap': np.arange(1, 11), 'Cw_over_Csat': r['Cw']/Csat, 'medium_entering_over_Csat': r['Cb_in']/Csat, 'J_mol_per_s': r['J']})
        if df is None: df = pd.DataFrame({'position_mm': x})
        df[f'medium C/Csat, {vname}'] = r['C']/Csat
    xl[f'Fig2_{lab}_medium_profile'] = df
    ax.set_xlabel('Position along channel (mm)'); ax.set_xlim(0, Lmm); ax.axhline(0.10, color='k', lw=0.6, ls=(0, (1, 2)))
    sec = ax.secondary_xaxis('top', functions=(lambda x: x/Lmm*10+0.5, lambda t: (t-0.5)*Lmm/10)); sec.set_xticks(range(1, 11)); sec.set_xlabel('Trap', labelpad=2)
axes[0].set_ylabel('O₂ (fraction of C_sat)'); axes[0].set_ylim(-0.02, 1.05); axes[0].set_title('LF, 1 µl/min', loc='left', pad=22)
axes[1].set_ylim(0.8, 1.02); axes[1].set_title('HF, 1,000 µl/min', loc='left', pad=22)
axes[0].text(Lmm*0.98, 0.72, 'dotted line: supplied\nthreshold 0.10 C_sat', ha='right', va='center', fontsize=6, linespacing=1.3)
qs = np.logspace(-17, math.log10(2e-16), 25); dfc = pd.DataFrame({'q_mol_per_cell_s': qs})
for vname, kw in models:
    sup = [E.summarize(E.trap_profile(P, P['Q_LF'], q, Csat, **kw), P, q)[0] for q in qs]; dfc[vname] = sup; bound = not kw['transfer_limited']
    axes[2].step(qs, sup, where='mid', color=G2 if bound else LF, ls=':' if bound else '-', lw=1.1 if bound else 1.6)
xl['Fig2c_LF_traps_supplied_vs_q'] = dfc
axes[2].set_xscale('log'); axes[2].set_xlabel('O₂ uptake q (mol cell⁻¹ s⁻¹)'); axes[2].set_ylabel('LF traps with C_w ≥ 0.1 C_sat'); axes[2].set_ylim(-0.3, 11.5); axes[2].set_yticks(range(0, 11, 2))
for q, t in ((P['q']['low'], 'low'), (qc, 'central'), (P['q']['high'], 'high')):
    axes[2].axvline(q, color=G3, lw=0.6, zorder=0); axes[2].text(q, 10.6, t, ha='center', va='bottom', fontsize=6, color=G2)
axes[2].set_title('LF, supplied traps vs uptake', loc='left', pad=22)
hh, ll = axes[0].get_legend_handles_labels()
fig.legend(hh, ll, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16), fontsize=6.5, columnspacing=1.2)
for ax, s in zip(axes, 'abc'): letter(ax, s)
fig.savefig('fig_ext1_O2_profile_models.png', dpi=300, bbox_inches='tight'); fig.savefig('fig_ext1_O2_profile_models.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 3: sensitivity ----------------
def evaluate(p, q=None, Sh=None, Aface=None, LBscale=1.0, model_kw=dict(transfer_limited=True, Km=1e-3)):
    q = p['q']['central'] if q is None else q; res = {}
    r = E.trap_profile(p, p['Q_LF'], q, Csat, Sh=Sh, Aface=Aface, **model_kw)
    res['LF traps supplied (of 10)'], res['LF demand met'] = E.summarize(r, p, q); res['LF outlet C/Csat'] = r['Cb_out'][-1]/Csat
    r = E.trap_profile(p, p['Q_HF'], q, Csat, Sh=Sh, Aface=Aface, **model_kw)
    res['HF traps supplied (of 10)'], res['HF demand met by face diffusion'] = E.summarize(r, p, q); res['HF C_wall/Csat trap 1'] = r['Cw'][0]/Csat
    c = channel(p, p['Q_HF'])
    res['HF interstitial O2 / trap demand, trap 1'] = p['kappa']*c['dP1']*LBscale/(p['mu']*p['LT'])*p['wT']*p['h']*Csat/(p['N']/p['ntrap']*q)
    res['HF capillary tau trap 1 (dyn/cm2)'] = c['dP1']*LBscale*p['d_cap']/(4*p['Lv'])*10
    return res
SWEEPS = {
 'κ 10⁻¹⁵ / 10⁻¹³ m²': [dict(p_over=dict(kappa=1e-15)), dict(p_over=dict(kappa=1e-13))],
 'q 2×10⁻¹⁷ / 10⁻¹⁶ mol s⁻¹ per cell': [dict(q=2e-17), dict(q=1e-16)],
 'N 10⁵ / 10⁶ cells': [dict(p_over=dict(N=1e5, nc=P['nc']/5)), dict(p_over=dict(N=1e6, nc=P['nc']*2))],
 'L_B(n) ×0.7 / ×1.3': [dict(LBscale=0.7), dict(LBscale=1.3)],
 'Sh at LF 2 / 6': [dict(Sh=2.0), dict(Sh=6.0)],
 'interface area 0.4 / 0.5 mm²': [dict(Aface=4e-7), dict(Aface=5e-7)],
 'hypoxic down-regulation α 0 / 0.9': [dict(model_kw=dict(transfer_limited=True, Km=1e-3, alpha=0.0)), dict(model_kw=dict(transfer_limited=True, Km=1e-3, alpha=0.9))],
}
base = evaluate(P); rows = []
for name, (lo, hi) in SWEEPS.items():
    vals = []
    for side in (lo, hi):
        p2 = copy.deepcopy(P); p2.update(side.get('p_over', {})); vals.append(evaluate(p2, **{k: v for k, v in side.items() if k != 'p_over'}))
    for out in base: rows.append(dict(parameter=name, output=out, baseline=base[out], low=vals[0][out], high=vals[1][out]))
sens = pd.DataFrame(rows); xl['Fig3_sensitivity_full'] = sens
outs = [('LF traps supplied (of 10)', 'LF traps supplied (of 10)', LF, False, (0, 10.5)),
        ('HF demand met by face diffusion', 'HF: fraction of demand met\nby face diffusion', HF, False, (0, 1.0)),
        ('HF interstitial O2 / trap demand, trap 1', 'HF: interstitial O₂ supply\n÷ trap-1 demand', HF, True, (0.005, 3))]
order = list(SWEEPS.keys())
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.25))
for ax, (key, lab, col, log, xlim) in zip(axes, outs):
    sub = sens[sens.output == key].set_index('parameter').loc[order]; yy = np.arange(len(order))[::-1]; b = base[key]
    for y_, (pn, r) in zip(yy, sub.iterrows()):
        if abs(r.high - r.low) < 1e-9: ax.plot([b], [y_], '|', color=G1, ms=6); continue
        ax.plot([r.low, r.high], [y_, y_], color=col, lw=5, solid_capstyle='butt', alpha=0.85); ax.plot([r.low], [y_], marker=4, color='k', ms=4, mew=0.8, mfc='none')
    ax.axvline(b, color='k', lw=0.7)
    if log:
        ax.set_xscale('log'); ax.axvline(1, color=G1, lw=0.6, ls=(0, (1, 2))); ax.text(1, len(order)-0.5, 'demand met', ha='center', va='bottom', fontsize=6, color=G2)
        ax.set_xticks([0.01, 0.1, 1]); ax.set_xticklabels(['0.01', '0.1', '1'])
    ax.set_xlim(*xlim); ax.set_yticks(yy); ax.set_yticklabels(order if ax is axes[0] else [], fontsize=6.5); ax.set_xlabel(lab, fontsize=6.5); ax.set_ylim(-0.6, len(order)-0.4)
fig.suptitle("Bar: output from the low to the high value of one parameter; vertical line: baseline; ◄ marks the parameter's low end", x=0.5, y=1.02, fontsize=7)
for ax, s in zip(axes, 'abc'): letter(ax, s)
fig.savefig('fig_ext3_sensitivity_tornado.png', dpi=300, bbox_inches='tight'); fig.savefig('fig_ext3_sensitivity_tornado.pdf', bbox_inches='tight'); plt.close(fig)

with pd.ExcelWriter('VNO_note_v3_raw_data.xlsx') as w:
    for k, v in xl.items(): v.to_excel(w, sheet_name=k[:31], index=False)
if __name__ == '__main__':
    t = xl['Fig2_LF_tissue_surface']; print('LF Cw/Csat trap 1..10:', np.round(t.Cw_over_Csat.values, 2))
    print('HF Cw/Csat trap 1:', round(xl['Fig2_HF_tissue_surface'].Cw_over_Csat.iloc[0], 3))
    print(pd.Series(base).round(3).to_string())
