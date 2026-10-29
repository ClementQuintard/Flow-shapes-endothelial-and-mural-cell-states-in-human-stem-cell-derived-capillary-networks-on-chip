"""Figure 3 panels d, e, f for the open 500 x 500 um channel, LF 1 / HF 1,000 ul/min, 5e4 cells per trap.
Panel f uses the transport-limited per-trap balance of vno_extensions.py (not the zero-order cliff).
Run from the physics package directory (needs vno_model.py and vno_extensions.py): python make_fig3_panels.py"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import vno_extensions as E
from vno_model import channel, ul
P, Csat = E.P, E.Csat; N, ntrap, q = P['N'], P['ntrap'], P['q']
Qs = {'LF': P['Q_LF'], 'HF': P['Q_HF']}
LFC, HFC, GREY, INK, INK2 = '#1f4e9c', '#00b8d9', '#8c8b85', '#1f1f1d', '#5f5e59'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.8})
fmt = FuncFormatter(lambda v, _: f'{v:g}')
n = np.arange(1, ntrap + 1)
LB = P['LB1'] - (P['LB1'] - P['LB10'])/(ntrap - 1)*(n - 1)          # 40 -> 13 mm, linear in n
sheets = {}

# ---- panel d: shear by trap ----
rows = []
for c, Qul in Qs.items():
    ch = channel(P, Qul); dP = ch['G']*LB; tcap = dP*P['d_cap']/(4*P['Lv'])*10
    for i in range(ntrap):
        rows.append({'Condition': c, 'Q (ul/min)': Qul, 'Trap position': n[i], 'L_B (mm)': LB[i]*1e3, 'Delta P (Pa)': dP[i],
                     'Channel wall shear, mean (dyn/cm2)': ch['tau']*10, 'Channel wall shear, mid-wall peak (dyn/cm2)': ch['tau_mid']*10,
                     'Capillary wall shear, 10 um (dyn/cm2)': tcap[i]})
d = pd.DataFrame(rows); sheets['d_shear_by_trap'] = d
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.axhspan(1, 6, color=GREY, alpha=0.10, lw=0); ax.text(10.4, 2.4, 'veins\n1–6', color=INK2, fontsize=7, va='center')
ax.axhspan(10, 70, color=GREY, alpha=0.18, lw=0); ax.text(10.4, 26, 'arteries\n10–70', color=INK2, fontsize=7, va='center')
for c, col in [('HF', HFC), ('LF', LFC)]:
    s = d[d.Condition == c]
    ax.plot(s['Trap position'], s['Capillary wall shear, 10 um (dyn/cm2)'], '-o', color=col, lw=2, ms=4, label=f'{c}, capillaries')
    ax.plot(s['Trap position'], s['Channel wall shear, mean (dyn/cm2)'], '--', color=col, lw=1.5, label=f'{c}, channel wall')
ax.set_yscale('log'); ax.set_ylim(0.001, 100); ax.set_xlim(0.5, 10.3); ax.set_xticks(n); ax.yaxis.set_major_formatter(fmt)
ax.set_xlabel('Trap position'); ax.set_ylabel('Wall shear stress (dyn/cm²)')
ax.set_title('Shear under HF reaches the venous to arterial range', fontsize=9, loc='left', color=INK)
ax.legend(frameon=False, fontsize=6.5, loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2)
fig.tight_layout(); fig.savefig('fig3d_shear.png', dpi=300, bbox_inches='tight'); fig.savefig('fig3d_shear.pdf', bbox_inches='tight'); plt.close(fig)

# ---- panel e: Da vs Q ----
Qg = np.logspace(np.log10(0.5), np.log10(3000), 80)
dag = pd.DataFrame({'Q (ul/min)': Qg, **{f'Da ({k} uptake)': N*v/(ul(Qg)*Csat) for k, v in q.items()}}); sheets['e_Da_vs_Q'] = dag
fig, ax = plt.subplots(figsize=(3.8, 3.1))
ax.fill_between(Qg, dag['Da (low uptake)'], dag['Da (high uptake)'], color=GREY, alpha=0.25, lw=0, label='low–high uptake')
ax.plot(Qg, dag['Da (central uptake)'], color=INK2, lw=2, label='central uptake')
ax.axhline(1, color=INK2, ls='--', lw=1); ax.text(0.55, 1.25, 'Da = 1', color=INK2, fontsize=7)
ax.text(0.55, 40, 'O₂ supply-limited', color=INK2, fontsize=7); ax.text(0.55, 0.012, 'O₂-replete', color=INK2, fontsize=7)
pts = []
for c, col in [('LF', LFC), ('HF', HFC)]:
    Qv = Qs[c]; f = lambda v: N*v/(ul(Qv)*Csat); da, lo, hi = f(q['central']), f(q['low']), f(q['high'])
    pts.append({'Condition': c, 'Q (ul/min)': Qv, 'Da low': lo, 'Da central': da, 'Da high': hi})
    ax.errorbar(Qv, da, yerr=[[da-lo], [hi-da]], fmt='o', color=col, ms=6, capsize=3, lw=1.5, zorder=5)
    ax.text(Qv*1.3 if c == 'LF' else Qv/1.7, da, f'{c}\nDa {lo:.2g}–{hi:.2g}', color=INK, fontsize=7, va='center', ha='left' if c == 'LF' else 'right', ma='left' if c == 'LF' else 'right')
sheets['e_Da_points'] = pd.DataFrame(pts)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-3, 60); ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
ax.set_xlabel('Flow rate Q (µl/min)'); ax.set_ylabel('Damköhler number Da')
ax.set_title('LF is O₂-limited; HF is not', fontsize=9, loc='left', color=INK); ax.legend(frameon=False, fontsize=7, loc='upper right')
fig.tight_layout(); fig.savefig('fig3e_Da.png', dpi=300); fig.savefig('fig3e_Da.pdf'); plt.close(fig)

# ---- panel f: O2 reaching each trap, transport-limited model ----
Ty, Pt = P['Tygon'], P['PTFE']
o2 = []
for c, Qul in Qs.items():
    for k, v in q.items():
        cin = Csat        # inlet taken at air saturation
        r = E.trap_profile(P, Qul, v, cin)
        for i in range(ntrap):
            o2.append({'Condition': c, 'Q (ul/min)': Qul, 'Uptake': k, 'Trap position': n[i], 'Inlet C/Csat': cin/Csat,
                       'O2 in medium entering trap (fraction of saturation)': r['Cb_in'][i]/Csat,
                       'O2 at tissue surface (fraction of saturation)': r['Cw'][i]/Csat})
o2 = pd.DataFrame(o2); sheets['f_O2_by_trap'] = o2
fig, ax = plt.subplots(figsize=(3.8, 3.1))
for c, col in [('HF', HFC), ('LF', LFC)]:
    s = o2[o2.Condition == c].pivot(index='Trap position', columns='Uptake', values='O2 in medium entering trap (fraction of saturation)')
    ax.fill_between(s.index, s['high'], s['low'], color=col, alpha=0.18, lw=0)
    ax.plot(s.index, s['central'], '-o', color=col, lw=2, ms=4, label=f'{c} ({Qs[c]:,.0f} µl/min)')
ax.axhline(0.10, color=INK2, ls=':', lw=0.8); ax.text(0.7, 0.12, '10% of saturation', color=INK2, fontsize=6.5, ha='left', va='bottom')
ax.set_ylim(-0.02, 1.08); ax.set_xticks(n); ax.set_xlabel('Trap position'); ax.set_ylabel('O₂ entering trap (fraction of saturation)')
ax.set_title('Under LF, O₂ declines trap by trap', fontsize=9, loc='left', color=INK)
ax.legend(frameon=False, fontsize=8, loc='center right'); ax.text(10.3, 1.03, 'line: central uptake; band: low–high', color=INK2, fontsize=6.5, ha='right')
fig.tight_layout(); fig.savefig('fig3f_O2_by_trap.png', dpi=300); fig.savefig('fig3f_O2_by_trap.pdf'); plt.close(fig)

with pd.ExcelWriter('Fig3_def_raw_data.xlsx') as w:
    for k, v in sheets.items(): v.to_excel(w, sheet_name=k, index=False)
if __name__ == '__main__':
    pd.set_option('display.width', 200)
    print(d.groupby('Condition')[['Channel wall shear, mean (dyn/cm2)', 'Capillary wall shear, 10 um (dyn/cm2)']].agg(['first', 'last']).to_string())
    print(pd.DataFrame(pts).to_string())
    print(o2.pivot_table(index='Trap position', columns=['Condition', 'Uptake'], values='O2 in medium entering trap (fraction of saturation)').round(2).to_string())
