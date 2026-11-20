"""
Simulateur de chromatographe ionique (IC)
Stack : Streamlit, numpy, scipy, plotly
Lancement : streamlit run simulateur_ic.py

Modèle (pédagogique, valeurs typiques de colonnes 250 x 4 mm) :
- rétention par échange d'ions : k' = k'ref * (E/Eref)^(-z/z_éluant) * f(T)
- tR = t0 * (1 + k'), t0 = volume mort / débit
- pics = gaussiennes modifiées exponentiellement (EMG), largeur sigma = tR / sqrt(N)
- aire (µS·min) = Λ_eq * z * c(mmol/L) * V_inj / débit
- détection de pics et mesures (tR, aire, W1/2, asymétrie, N, Rs) par scipy
"""
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

# tr  : temps de rétention de référence (min) : 250 mm, éluant/débit de référence, 30 °C
# z   : charge | lam : conductivité équivalente limite (S·cm²/eq) | M : g/mol
# alpha : fraction ionisée après suppression (acides faibles) | c0 : concentration par défaut (mg/L)
ANIONS = {
    "F⁻":      dict(tr=2.8,  z=1, lam=54.4, M=19.00, alpha=0.6, c0=2),
    "Cl⁻":     dict(tr=4.2,  z=1, lam=76.3, M=35.45, alpha=1.0, c0=10),
    "NO₂⁻":    dict(tr=5.0,  z=1, lam=71.8, M=46.01, alpha=0.7, c0=5),
    "Br⁻":     dict(tr=6.3,  z=1, lam=78.1, M=79.90, alpha=1.0, c0=10),
    "NO₃⁻":    dict(tr=7.2,  z=1, lam=71.4, M=62.00, alpha=1.0, c0=10),
    "HPO₄²⁻":  dict(tr=9.5,  z=2, lam=57.0, M=95.00, alpha=0.4, c0=15),
    "SO₄²⁻":   dict(tr=10.8, z=2, lam=80.0, M=96.06, alpha=1.0, c0=20),
}
CATIONS = {
    "Li⁺":     dict(tr=3.5,  z=1, lam=38.7, M=6.94,  alpha=1.0, c0=1),
    "Na⁺":     dict(tr=4.3,  z=1, lam=50.1, M=22.99, alpha=1.0, c0=10),
    "NH₄⁺":    dict(tr=5.0,  z=1, lam=73.5, M=18.04, alpha=1.0, c0=2),
    "K⁺":      dict(tr=6.5,  z=1, lam=73.5, M=39.10, alpha=1.0, c0=5),
    "Mg²⁺":    dict(tr=9.0,  z=2, lam=53.0, M=24.31, alpha=1.0, c0=5),
    "Ca²⁺":    dict(tr=11.0, z=2, lam=59.5, M=40.08, alpha=1.0, c0=10),
}
MODES = {
    "Anions": dict(ions=ANIONS, E_ref=8.0, E_range=(2.0, 30.0), ze=2, flow_ref=1.2,
                   lam_sup=349.8, lam_direct=38.0, bg=1.2,
                   eluant="carbonate/bicarbonate"),
    "Cations": dict(ions=CATIONS, E_ref=20.0, E_range=(5.0, 60.0), ze=1, flow_ref=1.0,
                    lam_sup=198.6, lam_direct=349.8, bg=0.3,
                    eluant="acide méthanesulfonique"),
}


# ------------------------------------------------------------------ Modèle
def v0(L):
    """Volume mort (mL) pour une colonne de longueur L (mm)."""
    return np.pi * (ID_MM / 20) ** 2 * (L / 10) * POROSITE


def retention(ion, E, debit, T, L, cfg):
    """Retourne (tR, t0, k')."""
    t0_ref = v0(L_REF) / cfg["flow_ref"]
    k_ref = ion["tr"] / t0_ref - 1
    k = k_ref * (E / cfg["E_ref"]) ** (-ion["z"] / cfg["ze"]) * np.exp(-0.015 * (T - 30))
    t0 = v0(L) / debit
    return t0 * (1 + k), t0, k


def aire_pic(ion, c_mgl, v_inj_ul, debit, cfg, supprime):
    """Aire théorique (µS·min). Conductivité supprimée : H+ (anions) ou OH- (cations)
    remplace le contre-ion ; directe : signal = différence avec l'ion d'éluant."""
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


def simuler(echantillon, ions, cfg, E, debit, T, L, N250, r, v_inj,
            supprime, bruit, derive, hz, duree, vide, seed):
    dt = 1 / (60 * hz)
    t = np.arange(0, duree, dt)
    y = cfg["bg"] + derive * t
    t0 = v0(L) / debit
    N = N250 * L / L_REF
    for nom, c in echantillon.items():
        ion = ions[nom]
        tr = retention(ion, E, debit, T, L, cfg)[0]
        sigma = tr / np.sqrt(N)
        y = y + aire_pic(ion, c, v_inj, debit, cfg, supprime) * emg(t, tr, sigma, r)
    if vide:  # perturbation d'injection (eau) au volume mort
        y = y - 0.02 * (v_inj / 25) * emg(t, t0, 0.05, 0.0)
    rng = np.random.default_rng(int(seed))
    y = y + rng.normal(0, bruit, t.size)
    return t, y, t0


# ------------------------------------------------------------------ Analyse
def baseline_fit(t, y):
    """Ligne de base par écrêtage itératif (SNIP) + écart-type du bruit.
    La polarité dominante des pics (positifs en supprimé, négatifs en direct
    pour les cations) est détectée automatiquement."""
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
    """Détecte et mesure les pics ; ignore les artefacts voisins du volume mort t0."""
    dt = t[1] - t[0]
    base, bruit = baseline_fit(t, y)
    yc = y - base
    win = max(7, int(0.03 / dt) // 2 * 2 + 1)
    ys = savgol_filter(yc, win, 3)
    ya = np.abs(ys)
    pics, _ = find_peaks(ya, prominence=snr_min * bruit)
    if len(pics) == 0:
        return pd.DataFrame()
    w50 = peak_widths(ya, pics, rel_height=0.5)
    w10 = peak_widths(ya, pics, rel_height=0.9)
    w98 = peak_widths(ya, pics, rel_height=0.98)
    lignes = []
    for i, ip in enumerate(pics):
        a_i, b_i = int(np.floor(w98[2][i])), int(np.ceil(w98[3][i]))
        fwhm = w50[0][i] * dt
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
    # artefacts de ligne de base autour de t0 (perturbation d'injection)
    df = df[~((df["Attribution"] == "?") & ((df["tR (min)"] - t0).abs() < 0.4))].reset_index(drop=True)
    if df.empty:
        return df
    df["Pic"] = np.arange(1, len(df) + 1)
    rs = [np.nan] + [
        1.18 * (df.loc[i, "tR (min)"] - df.loc[i - 1, "tR (min)"])
        / (df.loc[i, "W½ (min)"] + df.loc[i - 1, "W½ (min)"])
        for i in range(1, len(df))
    ]
    df.insert(7, "Rs", rs)
    return df


# ------------------------------------------------------------------ Interface
st.title("Simulateur de chromatographe ionique")

with st.sidebar:
    st.header("Paramètres")
    mode = st.radio("Mode", list(MODES), horizontal=True)
    cfg = MODES[mode]
    supprime = st.toggle("Conductivité supprimée", value=True)

    st.subheader("Séparation")
    E = st.slider(f"Éluant {cfg['eluant']} (mM)", cfg["E_range"][0], cfg["E_range"][1],
                  cfg["E_ref"], 0.5)
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

ions = cfg["ions"]
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

lib = {n: retention(i, E, debit, T, L, cfg)[0] for n, i in ions.items()}
t, y, t0 = simuler(echantillon, ions, cfg, E, debit, T, L, N250, r, v_inj,
                   supprime, bruit, derive, hz, duree, vide, seed)
tab = analyser(t, y, snr_min, lib, t0)

tab1, tab2, tab3 = st.tabs(["Chromatogramme", "Pics détectés", "Rétention vs éluant"])

with tab1:
    hors = [n for n in echantillon if lib[n] > duree]
    if hors:
        st.warning(f"Hors fenêtre d'analyse : {', '.join(hors)}. Augmente la durée ou l'éluant.")
    fig = go.Figure(go.Scatter(x=t, y=y, mode="lines", line=dict(width=1.3), name="Signal"))
    fig.add_vline(x=t0, line_dash="dot", annotation_text="t₀")
    if not tab.empty:
        for _, p in tab.iterrows():
            fig.add_annotation(x=p["tR (min)"], y=y[int(p["_i"])], text=p["Attribution"],
                               showarrow=True, arrowhead=0, ax=0,
                               ay=-30 if p["Hauteur (µS/cm)"] > 0 else 30)
    fig.update_layout(xaxis_title="Temps (min)", yaxis_title="Conductivité (µS/cm)",
                      height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(fig)
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
    Es = np.geomspace(cfg["E_range"][0], cfg["E_range"][1], 60)
    fig2 = go.Figure()
    for n in (echantillon or ions):
        fig2.add_trace(go.Scatter(
            x=Es, y=[retention(ions[n], e, debit, T, L, cfg)[0] for e in Es],
            mode="lines", name=n))
    fig2.add_vline(x=E, line_dash="dot")
    fig2.update_xaxes(type="log", title="Concentration d'éluant (mM)")
    fig2.update_yaxes(title="tR (min)")
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2)
    st.caption("Les ions divalents sont plus sensibles à l'éluant que les monovalents : "
               "log k' varie en −(z / z_éluant) · log [E].")
