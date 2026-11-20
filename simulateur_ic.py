"""
Simulateur de chromatographe ionique (IC) avec analyse quantitative
Stack : Streamlit, numpy, pandas, scipy, plotly
Lancement : streamlit run simulateur_ic.py

Modèle (pédagogique, valeurs typiques de colonnes 250 x 4 mm) :
- anions : éluant carbonate (Na2CO3) + hydrogénocarbonate (NaHCO3) réglés séparément.
  k' = k'ref * (B/Bref)^(-fB*z) * (C/Cref)^(-(1-fB)*z/2)
  (B, C : mM d'hydrogénocarbonate et de carbonate ; z : charge de l'analyte ;
   fB : part de l'hydrogénocarbonate dans la force d'élution de référence, en équivalents)
- cations : éluant acide méthanesulfonique, k' = k'ref * (E/Eref)^(-z)
- tR = t0 * (1 + k'), t0 = volume mort / débit
- pics = gaussiennes modifiées exponentiellement (EMG), sigma = tR / sqrt(N)
- aire (µS·min) = Λ_eq * z * c(mmol/L) * V_inj / débit
- détection de pics et mesures (tR, aire, W1/2, asymétrie, N, Rs) par scipy
- étalonnage : droite aire = f(concentration) pour chaque anion, puis dosage des échantillons
"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.special import erfcx

st.set_page_config(page_title="Simulateur de chromatographe ionique", layout="wide")

# ------------------------------------------------------------------ Constantes
ID_MM = 4.0       # diamètre interne de la colonne (mm)
POROSITE = 0.40   # fraction de volume mort
L_REF = 250.0     # longueur de référence (mm)
M_F = 19.00       # masse molaire du fluor (g/mol)

F_, CL, NO2, BR, NO3, MFP, SO4, PO4 = (
    "F⁻", "Cl⁻", "NO₂⁻", "Br⁻", "NO₃⁻", "FPO₃²⁻", "SO₄²⁻", "HPO₄²⁻")

# tr  : temps de rétention de référence (min) : 250 mm, éluant et débit de référence, 30 °C
# z   : charge | lam : conductivité équivalente limite (S·cm²/eq) | M : g/mol
# alpha : fraction ionisée après suppression (acides faibles) | c0 : concentration par défaut (mg/L)
# MFP (monofluorophosphate) : tR placé entre NO3⁻ et SO4²⁻ ; λ estimée.
ANIONS = {
    F_:  dict(tr=2.8,  z=1, lam=54.4, M=19.00, alpha=0.6, c0=2),
    CL:  dict(tr=4.2,  z=1, lam=76.3, M=35.45, alpha=1.0, c0=10),
    NO2: dict(tr=5.0,  z=1, lam=71.8, M=46.01, alpha=0.7, c0=5),
    BR:  dict(tr=6.3,  z=1, lam=78.1, M=79.90, alpha=1.0, c0=10),
    NO3: dict(tr=7.2,  z=1, lam=71.4, M=62.00, alpha=1.0, c0=10),
    MFP: dict(tr=8.8,  z=2, lam=55.0, M=97.97, alpha=1.0, c0=10),
    SO4: dict(tr=10.2, z=2, lam=80.0, M=96.06, alpha=1.0, c0=20),
    PO4: dict(tr=12.0, z=2, lam=57.0, M=95.98, alpha=0.4, c0=15),
}
CATIONS = {
    "Li⁺":  dict(tr=3.5,  z=1, lam=38.7, M=6.94,  alpha=1.0, c0=1),
    "Na⁺":  dict(tr=4.3,  z=1, lam=50.1, M=22.99, alpha=1.0, c0=10),
    "NH₄⁺": dict(tr=5.0,  z=1, lam=73.5, M=18.04, alpha=1.0, c0=2),
    "K⁺":   dict(tr=6.5,  z=1, lam=73.5, M=39.10, alpha=1.0, c0=5),
    "Mg²⁺": dict(tr=9.0,  z=2, lam=53.0, M=24.31, alpha=1.0, c0=5),
    "Ca²⁺": dict(tr=11.0, z=2, lam=59.5, M=40.08, alpha=1.0, c0=10),
}
MODES = {
    "Anions": dict(type="anions", ions=ANIONS, B_ref=1.4, C_ref=4.5,
                   B_range=(0.1, 10.0), C_range=(0.5, 20.0), flow_ref=1.2,
                   lam_sup=349.8, lam_direct=38.0),
    "Cations": dict(type="cations", ions=CATIONS, E_ref=20.0, E_range=(5.0, 60.0),
                    flow_ref=1.0, lam_sup=198.6, lam_direct=349.8, bg=0.3),
}

# Eaux minérales : composition d'étiquette / fiche producteur (mg/L).
# Abatilles : nitrates sous la limite de détection ; Volvic : fluorures non indiqués.
EAUX = {
    "Eau Abatilles": {CL: 150.0, SO4: 8.0},
    "Eau Volvic": {CL: 15.0, NO3: 7.3, SO4: 9.0},
}


# ------------------------------------------------------------------ Modèle
def v0(L):
    """Volume mort (mL) pour une colonne de longueur L (mm)."""
    return np.pi * (ID_MM / 20) ** 2 * (L / 10) * POROSITE


def retention(ion, el, debit, T, L, cfg):
    """Retourne (tR, t0, k'). el = {'B':.., 'C':..} (anions) ou {'E':..} (cations)."""
    t0_ref = v0(L_REF) / cfg["flow_ref"]
    k_ref = ion["tr"] / t0_ref - 1
    if cfg["type"] == "anions":
        fB = cfg["B_ref"] / (cfg["B_ref"] + 2 * cfg["C_ref"])
        k = (k_ref
             * (el["B"] / cfg["B_ref"]) ** (-fB * ion["z"])
             * (el["C"] / cfg["C_ref"]) ** (-(1 - fB) * ion["z"] / 2))
    else:
        k = k_ref * (el["E"] / cfg["E_ref"]) ** (-ion["z"])
    k *= np.exp(-0.015 * (T - 30))
    t0 = v0(L) / debit
    return t0 * (1 + k), t0, k


def fond(el, cfg):
    """Conductivité de fond (µS/cm) après suppression."""
    if cfg["type"] == "anions":
        ct = (el["B"] + el["C"]) * 1e-3          # carbone inorganique total (mol/L)
        return 394000 * np.sqrt(4.45e-7 * ct)    # H2CO3 -> H+ + HCO3-
    return cfg["bg"]


def aire_pic(ion, c_mgl, v_inj_ul, debit, cfg, supprime):
    """Aire théorique (µS·min). Supprimée : H+ (anions) ou OH- (cations) remplace le
    contre-ion ; directe : signal = différence avec l'ion d'éluant."""
    c_mmol = c_mgl / ion["M"]
    if supprime:
        lam, f = ion["lam"] + cfg["lam_sup"], ion["alpha"]
    else:
        lam, f = ion["lam"] - cfg["lam_direct"], 1.0
    return lam * ion["z"] * f * c_mmol * (v_inj_ul / 1000) / debit


def emg(t, tr, sigma, r):
    """Gaussienne modifiée exponentiellement, aire normalisée à 1 (r = tau/sigma)."""
    x = t - tr
    if r < 0.05:
        return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    tau = r * sigma
    out = np.zeros_like(t)
    m = (x > -6 * sigma) & (x < 6 * sigma + 8 * tau)
    xm = x[m]
    b = (sigma / tau - xm / sigma) / np.sqrt(2)
    out[m] = np.exp(-xm ** 2 / (2 * sigma ** 2)) * erfcx(b) / (2 * tau)
    return out


def bibliotheque(P):
    """tR théorique de chaque ion aux conditions courantes (sert à l'attribution)."""
    cfg = P["cfg"]
    return {n: retention(i, P["el"], P["debit"], P["T"], P["L"], cfg)[0]
            for n, i in cfg["ions"].items()}


def simuler(ech, P, seed):
    """Simule un chromatogramme. ech = {ion: mg/L injectés}. Retourne (t, y, t0)."""
    cfg = P["cfg"]
    ions = cfg["ions"]
    dt = 1 / (60 * P["hz"])
    t = np.arange(0, P["duree"], dt)
    y = fond(P["el"], cfg) + P["derive"] * t
    t0 = v0(P["L"]) / P["debit"]
    N = P["N250"] * P["L"] / L_REF
    for nom, c in ech.items():
        ion = ions[nom]
        tr = retention(ion, P["el"], P["debit"], P["T"], P["L"], cfg)[0]
        sigma = tr / np.sqrt(N)
        y = y + aire_pic(ion, c, P["v_inj"], P["debit"], cfg, P["supprime"]) * emg(t, tr, sigma, P["r"])
    if P["vide"]:  # perturbation d'injection (eau) au volume mort
        y = y - 0.02 * (P["v_inj"] / 25) * emg(t, t0, 0.05, 0.0)
    rng = np.random.default_rng(int(seed))
    y = y + rng.normal(0, P["bruit"], t.size)
    return t, y, t0


# ------------------------------------------------------------------ Analyse
def baseline_fit(t, y):
    """Ligne de base par écrêtage itératif (SNIP) + écart-type du bruit.
    La polarité dominante des pics est détectée automatiquement."""
    dt = t[1] - t[0]
    noise = max(1.4826 * np.median(np.abs(np.diff(y) - np.median(np.diff(y)))) / np.sqrt(2), 1e-4)
    win = max(7, int(0.1 / dt) // 2 * 2 + 1)
    v = savgol_filter(y, win, 3)
    dev = v - np.median(v)
    sign = 1.0 if dev.max() >= -dev.min() else -1.0
    w = sign * v
    m = min(int(1.2 / dt), (len(w) - 1) // 2 - 1)
    for p in range(m, 0, -1):
        w[p:-p] = np.minimum(w[p:-p], 0.5 * (w[:-2 * p] + w[2 * p:]))
    return sign * w, noise


def identifier(tr, lib):
    if not lib:
        return "?"
    nom, tl = min(lib.items(), key=lambda kv: abs(kv[1] - tr))
    return nom if abs(tl - tr) <= max(0.12, 0.03 * tr) else "?"


def analyser(t, y, snr_min, lib, t0):
    """Détecte et mesure les pics. Retourne (tableau, bruit). Les artefacts de ligne de
    base voisins du volume mort t0 sont ignorés."""
    dt = t[1] - t[0]
    base, bruit = baseline_fit(t, y)
    yc = y - base
    win = max(7, int(0.03 / dt) // 2 * 2 + 1)
    ys = savgol_filter(yc, win, 3)
    ya = np.abs(ys)
    pics, _ = find_peaks(ya, prominence=snr_min * bruit)
    if len(pics) == 0:
        return pd.DataFrame(), bruit
    w50 = peak_widths(ya, pics, rel_height=0.5)
    w10 = peak_widths(ya, pics, rel_height=0.9)
    w98 = peak_widths(ya, pics, rel_height=0.98)
    lignes = []
    for i, ip in enumerate(pics):
        fwhm = w50[0][i] * dt
        # fenêtre d'intégration bornée autour du pic (évite d'intégrer du bruit
        # quand le pic est isolé) : de -2,5 à +4 largeurs à mi-hauteur
        a_i = max(int(np.floor(w98[2][i])), ip - int(2.5 * fwhm / dt), 0)
        b_i = min(int(np.ceil(w98[3][i])), ip + int(4 * fwhm / dt), len(t) - 1)
        a, b = (ip - w10[2][i]) * dt, (w10[3][i] - ip) * dt
        lignes.append({
            "Pic": i + 1,
            "Attribution": identifier(t[ip], lib),
            "tR (min)": t[ip],
            "Hauteur (µS/cm)": ys[ip],
            "Aire (µS·min)": trapezoid(yc[a_i:b_i + 1], t[a_i:b_i + 1]),
            "W½ (min)": fwhm,
            "As (10 %)": b / a if a > 0 else np.nan,
            "N (plateaux)": 5.54 * (t[ip] / fwhm) ** 2,
            "S/B": abs(ys[ip]) / bruit,
            "_i": ip,
        })
    df = pd.DataFrame(lignes)
    df = df[~((df["Attribution"] == "?") & ((df["tR (min)"] - t0).abs() < 0.4))].reset_index(drop=True)
    if df.empty:
        return df, bruit
    df["Pic"] = np.arange(1, len(df) + 1)
    rs = [np.nan] + [
        1.18 * (df.loc[i, "tR (min)"] - df.loc[i - 1, "tR (min)"])
        / (df.loc[i, "W½ (min)"] + df.loc[i - 1, "W½ (min)"])
        for i in range(1, len(df))
    ]
    df.insert(7, "Rs", rs)
    return df, bruit


# ------------------------------------------------------------------ Analyse quantitative
def parse_niveaux(txt):
    """'5 10 20' ou '5; 10; 20' -> [5.0, 10.0, 20.0] (virgule ou point décimal)."""
    vals = []
    for tok in re.split(r"[;\s]+", txt.strip()):
        tok = tok.strip(",").replace(",", ".")
        try:
            v = float(tok)
        except ValueError:
            continue
        if v > 0:
            vals.append(v)
    return sorted(set(vals))[:8]


@st.cache_data(show_spinner=False)
def etalonnage(P, niveaux, melange, seed):
    """Simule les étalons (chaque anion seul, ou mélange) et ajuste les droites
    aire = a·c + b. Retourne (runs, cal)."""
    ions = P["cfg"]["ions"]
    lib = bibliotheque(P)
    groupes = {"Mélange": list(ions)} if melange else {n: [n] for n in ions}
    runs = {}
    for gi, (g, noms) in enumerate(groupes.items()):
        for j, c in enumerate(niveaux):
            t, y, t0 = simuler({n: c for n in noms}, P, seed + 1000 * gi + j)
            tab, bruit = analyser(t, y, P["snr"], lib, t0)
            runs[(g, c)] = dict(t=t, y=y, t0=t0, tab=tab, bruit=bruit)
    cal = {}
    for n in ions:
        g = "Mélange" if melange else n
        xs, aa, hh, bb = [], [], [], []
        for c in niveaux:
            r_ = runs[(g, c)]
            bb.append(r_["bruit"])
            tab = r_["tab"]
            sel = tab[tab["Attribution"] == n] if not tab.empty else tab
            if len(sel):
                row = sel.loc[sel["Hauteur (µS/cm)"].abs().idxmax()]
                xs.append(c)
                aa.append(row["Aire (µS·min)"])
                hh.append(row["Hauteur (µS/cm)"])
        if len(xs) < 2:
            cal[n] = None
            continue
        x, a_, h_ = np.array(xs), np.array(aa), np.array(hh)
        pa, ph = np.polyfit(x, a_, 1), np.polyfit(x, h_, 1)
        ss_res = ((a_ - np.polyval(pa, x)) ** 2).sum()
        ss_tot = ((a_ - a_.mean()) ** 2).sum()
        bruit_m = float(np.mean(bb))
        cal[n] = dict(x=x, aire=a_, haut=h_, a=float(pa[0]), b=float(pa[1]),
                      r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
                      lod=3 * bruit_m / abs(ph[0]), loq=10 * bruit_m / abs(ph[0]))
    return runs, cal


def quantifier(tab, cal, dil, attendu, niveaux):
    """Dose les pics d'un échantillon avec les droites d'étalonnage."""
    lignes, trouves = [], set()
    if not tab.empty:
        for _, p in tab.iterrows():
            n = p["Attribution"]
            if n not in cal or cal[n] is None:
                continue
            c_inj = (p["Aire (µS·min)"] - cal[n]["b"]) / cal[n]["a"]
            att = attendu.get(n, np.nan)
            if c_inj < min(niveaux):
                dom = "extrapolé (< étalon mini)"
            elif c_inj > max(niveaux):
                dom = "extrapolé (> étalon maxi)"
            else:
                dom = "dans la gamme"
            trouves.add(n)
            lignes.append({
                "Ion": n, "tR (min)": p["tR (min)"], "Aire (µS·min)": p["Aire (µS·min)"],
                "Trouvé, injecté (mg/L)": c_inj,
                "Dans l'échantillon (mg/L)": c_inj * dil,
                "Attendu (mg/L)": att,
                "Écart (%)": (c_inj * dil - att) / att * 100 if att and att > 0 else np.nan,
                "Domaine": dom,
            })
    for n, att in attendu.items():
        if n not in trouves:
            lignes.append({"Ion": n, "tR (min)": np.nan, "Aire (µS·min)": np.nan,
                           "Trouvé, injecté (mg/L)": np.nan, "Dans l'échantillon (mg/L)": np.nan,
                           "Attendu (mg/L)": att, "Écart (%)": np.nan, "Domaine": "non détecté"})
    return pd.DataFrame(lignes)


def fluor_total(dfq):
    """Fluor total retrouvé (ppm F) = F⁻ + MFP converti en fluor."""
    tot, ok = 0.0, False
    for _, r_ in dfq.iterrows():
        v = r_["Dans l'échantillon (mg/L)"]
        if pd.isna(v):
            continue
        if r_["Ion"] == F_:
            tot, ok = tot + v, True
        elif r_["Ion"] == MFP:
            tot, ok = tot + v * M_F / ANIONS[MFP]["M"], True
    return tot if ok else np.nan


# ------------------------------------------------------------------ Figures
def fig_chromato(t, y, tab, t0, titre=None, height=380):
    fig = go.Figure(go.Scatter(x=t, y=y, mode="lines", line=dict(width=1.3), name="Signal"))
    fig.add_vline(x=t0, line_dash="dot", annotation_text="t₀")
    if not tab.empty:
        for _, p in tab.iterrows():
            fig.add_annotation(x=p["tR (min)"], y=y[int(p["_i"])], text=p["Attribution"],
                               showarrow=True, arrowhead=0, ax=0,
                               ay=-30 if p["Hauteur (µS/cm)"] > 0 else 30)
    fig.update_layout(title=titre, xaxis_title="Temps (min)", yaxis_title="Conductivité (µS/cm)",
                      height=height, margin=dict(l=10, r=10, t=40 if titre else 20, b=10),
                      showlegend=False)
    return fig


def fig_etalons(g, titre, runs, niveaux):
    fig = go.Figure()
    for c in niveaux:
        r_ = runs[(g, c)]
        fig.add_trace(go.Scatter(x=r_["t"], y=r_["y"], mode="lines", name=f"{c:g} ppm",
                                 line=dict(width=1.2)))
    fig.update_layout(title=titre, xaxis_title="Temps (min)", yaxis_title="Conductivité (µS/cm)",
                      height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fig_cal(n, c_):
    xx = np.linspace(0, float(c_["x"].max()) * 1.05, 20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xx, y=c_["a"] * xx + c_["b"], mode="lines", name="Droite"))
    fig.add_trace(go.Scatter(x=c_["x"], y=c_["aire"], mode="markers", name="Étalons",
                             marker=dict(size=9)))
    fig.update_layout(title=f"{n} : A = {c_['a']:.4g}·c {c_['b']:+.3g}  (R² = {c_['r2']:.5f})",
                      xaxis_title="Concentration (mg/L)", yaxis_title="Aire (µS·min)",
                      height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


# ------------------------------------------------------------------ Interface
st.title("Simulateur de chromatographe ionique")

with st.sidebar:
    st.header("Paramètres")
    mode = st.radio("Mode", list(MODES), horizontal=True)
    cfg = MODES[mode]
    supprime = st.toggle("Conductivité supprimée", value=True)

    st.subheader("Séparation")
    if cfg["type"] == "anions":
        C = st.slider("Carbonate Na₂CO₃ (mM)", *cfg["C_range"], cfg["C_ref"], 0.1)
        B = st.slider("Hydrogénocarbonate NaHCO₃ (mM)", *cfg["B_range"], cfg["B_ref"], 0.1)
        el = {"B": B, "C": C}
    else:
        el = {"E": st.slider("Éluant acide méthanesulfonique (mM)", *cfg["E_range"],
                             cfg["E_ref"], 0.5)}
    debit = st.slider("Débit (mL/min)", 0.5, 2.0, cfg["flow_ref"], 0.05)
    T = st.slider("Température (°C)", 20, 50, 30)
    L = st.selectbox("Longueur de colonne (mm)", [100, 150, 250, 300], index=2)
    N250 = st.slider("Efficacité (plateaux / 250 mm)", 2000, 15000, 6000, 500)
    r = st.slider("Traînée des pics (τ/σ)", 0.0, 1.5, 0.4, 0.05)
    v_inj = st.slider("Volume injecté (µL)", 5, 100, 25, 5)

    st.subheader("Détecteur et analyse")
    bruit = st.slider("Bruit (µS/cm)", 0.0, 0.05, 0.003, 0.001)
    derive = st.slider("Dérive (µS/cm/min)", -0.2, 0.2, 0.02, 0.01)
    hz = st.select_slider("Fréquence d'acquisition (Hz)", [2, 5, 10], value=5)
    duree = st.slider("Durée d'analyse (min)", 5.0, 30.0, 15.0, 0.5)
    vide = st.checkbox("Pic de vide (injection)", value=True)
    snr_min = st.slider("Seuil de détection (S/B)", 3, 20, 5)
    seed = st.number_input("Graine du bruit", 0, 9999, 42)

P = dict(cfg=cfg, el=el, debit=debit, T=T, L=L, N250=N250, r=r, v_inj=v_inj,
         supprime=supprime, bruit=bruit, derive=derive, hz=hz, duree=duree,
         vide=vide, snr=snr_min)
ions = cfg["ions"]

if cfg["type"] == "anions":
    st.sidebar.caption(f"Na⁺ de l'éluant : {B + 2 * C:.1f} mM · fond supprimé ≈ {fond(el, cfg):.1f} µS/cm")

df0 = pd.DataFrame({
    "Ion": list(ions),
    "Actif": True,
    "mg/L": [float(v["c0"]) for v in ions.values()],
})
with st.expander("Échantillon (concentrations)"):
    ed = st.data_editor(
        df0, hide_index=True, key=f"ed_{mode}",
        column_config={
            "Ion": st.column_config.TextColumn(disabled=True),
            "Actif": st.column_config.CheckboxColumn(),
            "mg/L": st.column_config.NumberColumn(min_value=0.0, step=0.5),
        },
    )
echantillon = {row["Ion"]: float(row["mg/L"]) for _, row in ed.iterrows()
               if row["Actif"] and row["mg/L"] > 0}

lib = bibliotheque(P)
t, y, t0 = simuler(echantillon, P, seed)
tab, _ = analyser(t, y, snr_min, lib, t0)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Chromatogramme", "Pics détectés", "Rétention vs éluant", "Analyse quantitative"])

with tab1:
    hors = [n for n in echantillon if lib[n] > duree]
    if hors:
        st.warning(f"Hors fenêtre d'analyse : {', '.join(hors)}. Augmente la durée ou l'éluant.")
    st.plotly_chart(fig_chromato(t, y, tab, t0), key="chromato")
    st.caption(f"t₀ = {t0:.2f} min · {len(echantillon)} ion(s) injecté(s) · "
               f"{'détection supprimée' if supprime else 'détection directe'}")
    csv = pd.DataFrame({"Temps (min)": t, "Signal (µS/cm)": y}).to_csv(
        sep=";", decimal=",", index=False)
    st.download_button("Exporter le chromatogramme (CSV)", csv,
                       "chromatogramme_simule.csv", "text/csv")

with tab2:
    if tab.empty:
        st.info("Aucun pic détecté. Ajoute des ions ou baisse le seuil de détection.")
    else:
        aff = tab.drop(columns="_i").round(
            {"tR (min)": 3, "Hauteur (µS/cm)": 3, "Aire (µS·min)": 3, "W½ (min)": 3,
             "As (10 %)": 2, "Rs": 2, "N (plateaux)": 0, "S/B": 0})
        st.dataframe(aff, hide_index=True)
        st.caption("Rs < 1,5 : séparation incomplète · As > 1,2 : traînée · "
                   "attribution par comparaison au tR théorique de chaque ion.")

with tab3:
    if cfg["type"] == "anions":
        var = st.radio("Faire varier", ["Carbonate", "Hydrogénocarbonate"], horizontal=True)
        cle, plage, lab = (("C", cfg["C_range"], "Carbonate (mM)") if var == "Carbonate"
                           else ("B", cfg["B_range"], "Hydrogénocarbonate (mM)"))
    else:
        cle, plage, lab = "E", cfg["E_range"], "Éluant acide méthanesulfonique (mM)"
    Es = np.geomspace(*plage, 60)
    fig2 = go.Figure()
    for n in (echantillon or ions):
        fig2.add_trace(go.Scatter(
            x=Es, y=[retention(ions[n], {**el, cle: e}, debit, T, L, cfg)[0] for e in Es],
            mode="lines", name=n))
    fig2.add_vline(x=el[cle], line_dash="dot")
    fig2.update_xaxes(type="log", title=lab)
    fig2.update_yaxes(title="tR (min)")
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, key="retention")
    st.caption("Les ions divalents sont plus sensibles à l'éluant que les monovalents. "
               "Le carbonate (divalent) pèse davantage que l'hydrogénocarbonate dans la force d'élution.")

with tab4:
    if cfg["type"] != "anions":
        st.info("L'analyse quantitative (étalons, dentifrice, eaux) est disponible en mode Anions.")
    else:
        c1, c2 = st.columns(2)
        txt = c1.text_input("Niveaux d'étalonnage (ppm, séparés par des espaces)", "5 10 20")
        melange = c2.radio("Étalons", ["Chaque anion seul", "Mélange de tous les anions"]) \
            == "Mélange de tous les anions"
        niveaux = parse_niveaux(txt)
        if len(niveaux) < 2:
            st.warning("Indique au moins 2 niveaux (ex. : 5 10 20).")
        else:
            runs, cal = etalonnage(P, tuple(niveaux), melange, int(seed))
            sous = st.tabs(["Étalons", "Courbes d'étalonnage", "Échantillons"])

            with sous[0]:
                if melange:
                    st.plotly_chart(fig_etalons("Mélange", "Mélange de tous les anions", runs, niveaux),
                                    key="et_melange")
                else:
                    choix = st.selectbox("Anion", ["Tous"] + list(ions))
                    for n in (list(ions) if choix == "Tous" else [choix]):
                        st.plotly_chart(fig_etalons(n, f"{n} : " + ", ".join(f"{c:g}" for c in niveaux)
                                                    + " ppm", runs, niveaux), key=f"et_{n}")

            with sous[1]:
                lignes = []
                for n, c_ in cal.items():
                    if c_ is None:
                        lignes.append({"Ion": n})
                    else:
                        lignes.append({"Ion": n, "Pente (µS·min/ppm)": c_["a"],
                                       "Ordonnée (µS·min)": c_["b"], "R²": c_["r2"],
                                       "LD (ppm)": c_["lod"], "LQ (ppm)": c_["loq"]})
                st.dataframe(pd.DataFrame(lignes).round(5), hide_index=True)
                st.caption("LD = 3·bruit / pente (hauteurs), LQ = 10·bruit / pente.")
                cols = st.columns(2)
                for i, (n, c_) in enumerate(cal.items()):
                    if c_ is not None:
                        with cols[i % 2]:
                            st.plotly_chart(fig_cal(n, c_), key=f"cal_{n}")

            with sous[2]:
                a1, a2 = st.columns(2)
                f_tot = a1.number_input("Fluor du dentifrice (ppm F)", 100.0, 5000.0, 1450.0, 50.0)
                pct_mfp = a2.slider("Part du fluor sous forme MFP (%)", 0, 100, 90)
                d_dent = a1.number_input("Dilution du dentifrice (×)", 1, 5000, 200)
                d_aba = a2.number_input("Dilution Abatilles (×)", 1, 5000, 10)
                d_vol = a1.number_input("Dilution Volvic (×)", 1, 5000, 1)

                f_mfp = f_tot * pct_mfp / 100
                attendu_dent = {F_: f_tot - f_mfp, MFP: f_mfp * ANIONS[MFP]["M"] / M_F}
                echs = {
                    "Dentifrice": ({k: v for k, v in attendu_dent.items() if v > 0}, d_dent),
                    "Eau Abatilles": (EAUX["Eau Abatilles"], d_aba),
                    "Eau Volvic": (EAUX["Eau Volvic"], d_vol),
                }
                resultats = {}
                for i, (nom, (att, dil)) in enumerate(echs.items()):
                    inj = {n: v / dil for n, v in att.items()}
                    ts, ys, t0s = simuler(inj, P, int(seed) + 5000 + i)
                    tab_s, _ = analyser(ts, ys, snr_min, lib, t0s)
                    resultats[nom] = (ts, ys, t0s, tab_s, quantifier(tab_s, cal, dil, att, niveaux), dil)

                onglets = st.tabs(list(echs) + ["Superposition"])
                for k, nom in enumerate(echs):
                    ts, ys, t0s, tab_s, dfq, dil = resultats[nom]
                    with onglets[k]:
                        st.plotly_chart(fig_chromato(ts, ys, tab_s, t0s, f"{nom} (dilué ×{dil})"),
                                        key=f"ech_{nom}")
                        if not dfq.empty:
                            st.dataframe(dfq.round(3), hide_index=True)
                        if nom == "Dentifrice":
                            ft = fluor_total(dfq)
                            if not np.isnan(ft):
                                st.metric("Fluor total retrouvé (ppm F)", f"{ft:.0f}",
                                          f"{(ft - f_tot) / f_tot * 100:+.1f} % vs {f_tot:.0f}")
                with onglets[-1]:
                    fs = go.Figure()
                    for nom in echs:
                        fs.add_trace(go.Scatter(x=resultats[nom][0], y=resultats[nom][1],
                                                mode="lines", name=nom, line=dict(width=1.3)))
                    fs.update_layout(xaxis_title="Temps (min)", yaxis_title="Conductivité (µS/cm)",
                                     height=380, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fs, key="ech_sup")
                st.caption(
                    "Compositions (mg/L) : Abatilles Cl⁻ 150, SO₄²⁻ 8, nitrates sous la limite de détection ; "
                    "Volvic Cl⁻ 15, NO₃⁻ 7,3, SO₄²⁻ 9. Fluorures non indiqués : non simulés. "
                    "Un résultat « extrapolé » sort de la gamme d'étalonnage : ajoute des niveaux "
                    "(ex. 1 5 10 20 40).")
