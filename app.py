import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MLSecurity IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1220 100%);
    color: #c8d8e8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060c18 0%, #0a1220 100%) !important;
    border-right: 1px solid #1a3a5c;
}
section[data-testid="stSidebar"] * {
    color: #8ab4d0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #8ab4d0 !important;
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
section[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
    color: #8ab4d0 !important;
}

/* Titles */
h1 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #00d4ff !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
}
h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #4fc3f7 !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1e35 0%, #0a2040 100%);
    border: 1px solid #1a4a7a;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 20px rgba(0,180,255,0.08);
}
div[data-testid="metric-container"] label {
    color: #4fc3f7 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 26px !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1a3a5c !important;
    border-radius: 6px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #003d6b 0%, #005a9e 100%) !important;
    color: #00d4ff !important;
    border: 1px solid #0090d4 !important;
    border-radius: 6px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase;
    padding: 10px 24px !important;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #005a9e 0%, #0077cc 100%) !important;
    box-shadow: 0 0 20px rgba(0,180,255,0.3) !important;
    transform: translateY(-1px);
}

/* Selectbox & slider */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #0d1e35 !important;
    border: 1px solid #1a4a7a !important;
    border-radius: 6px !important;
    color: #8ab4d0 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #060c18;
    border-bottom: 1px solid #1a3a5c;
}
.stTabs [data-baseweb="tab"] {
    color: #4a7a9b !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Info/success boxes */
.stAlert {
    background: #0d1e35 !important;
    border: 1px solid #1a4a7a !important;
    border-radius: 6px !important;
    color: #8ab4d0 !important;
}

/* Section divider */
.section-divider {
    border: none;
    border-top: 1px solid #1a3a5c;
    margin: 20px 0;
}

/* Custom card */
.stat-card {
    background: linear-gradient(135deg, #0d1e35 0%, #0a2040 100%);
    border: 1px solid #1a4a7a;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #002244 0%, #003366 50%, #001a33 100%);
    border: 1px solid #004488;
    border-radius: 10px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,100,200,0.15);
}

/* Monospace values */
.mono {
    font-family: 'Share Tech Mono', monospace;
    color: #00d4ff;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess(df):
    data = df.copy()
    le = LabelEncoder()
    for col in ['protocol_type', 'service', 'flag']:
        if col in data.columns:
            data[col] = le.fit_transform(data[col])
    X = data.drop('labels', axis=1)
    y = data['labels']
    return X, y

MODELS = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}

COLORS = {
    "Decision Tree": "#4fc3f7",
    "Random Forest": "#00d4ff",
    "KNN": "#81d4fa",
    "Naive Bayes": "#4dd0e1",
    "Logistic Regression": "#26c6da",
}

ACCENT = "#00d4ff"
BG_DARK = "#0a0e1a"
BG_CARD = "#0d1e35"
GRID_COLOR = "#1a3a5c"
TEXT_COLOR = "#c8d8e8"


def apply_dark_theme(fig, ax_list=None):
    fig.patch.set_facecolor(BG_DARK)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(ACCENT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <span style='font-size:40px'>🛡️</span>
        <h2 style='color:#00d4ff; font-family:Rajdhani,sans-serif;
                   font-size:22px; letter-spacing:3px; margin:4px 0;'>
            MLSECURITY
        </h2>
        <p style='color:#4a7a9b; font-size:11px; letter-spacing:2px;
                  font-family:Share Tech Mono,monospace; margin:0;'>
            IDS · INTRUSION DETECTION
        </p>
    </div>
    <hr style='border-color:#1a3a5c; margin:12px 0;'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["🏠  Accueil",
         "📊  EDA & Exploration",
         "⚖️  Comparaison Modèles",
         "🌲  Feature Importance",
         "🔬  Test du Modèle",
         "💾  Sauvegarde Modèle"],
        label_visibility="visible"
    )

    st.markdown("<hr style='border-color:#1a3a5c; margin:16px 0;'/>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#4a7a9b; font-size:11px; font-family:Share Tech Mono,monospace;
              text-align:center; letter-spacing:1px;'>
        Dataset · KDD Cup 1999<br/>
        Algorithmes ML · Sklearn
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a3a5c; margin:8px 0;'/>", unsafe_allow_html=True)
    train_path = st.text_input("📂 Train CSV", value="dataset/kdd_train.csv")
    test_path = st.text_input("📂 Test CSV", value="dataset/kdd_test.csv")
    model_path = st.text_input("💾 Modèle (.pkl)", value="ids_model.pkl")


# ─────────────────────────────────────────────
# PAGE 0 — ACCUEIL
# ─────────────────────────────────────────────
if "Accueil" in page:
    st.markdown("""
    <div class='header-banner'>
        <h1 style='margin:0 0 6px 0; font-size:32px;'>🛡️ Système IDS — Détection d'Intrusions</h1>
        <p style='color:#4a9bc4; margin:0; font-family:Share Tech Mono,monospace; font-size:13px;
                  letter-spacing:1px;'>
            MACHINE LEARNING · SÉCURITÉ RÉSEAU · KDD CUP DATASET
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("""<div class='stat-card'>
        <p style='color:#4fc3f7; font-family:Rajdhani,sans-serif; font-weight:700;
                  font-size:13px; letter-spacing:1px; margin:0 0 6px 0;'>DATASET</p>
        <p style='color:#00d4ff; font-family:Share Tech Mono,monospace; font-size:22px; margin:0;'>
            KDD Cup 1999</p>
        <p style='color:#4a7a9b; font-size:12px; margin:4px 0 0 0;'>
            41 features · trafic réseau</p>
    </div>""", unsafe_allow_html=True)

    c2.markdown("""<div class='stat-card'>
        <p style='color:#4fc3f7; font-family:Rajdhani,sans-serif; font-weight:700;
                  font-size:13px; letter-spacing:1px; margin:0 0 6px 0;'>ALGORITHMES</p>
        <p style='color:#00d4ff; font-family:Share Tech Mono,monospace; font-size:22px; margin:0;'>
            5 Modèles</p>
        <p style='color:#4a7a9b; font-size:12px; margin:4px 0 0 0;'>
            DT · RF · KNN · NB · LR</p>
    </div>""", unsafe_allow_html=True)

    c3.markdown("""<div class='stat-card'>
        <p style='color:#4fc3f7; font-family:Rajdhani,sans-serif; font-weight:700;
                  font-size:13px; letter-spacing:1px; margin:0 0 6px 0;'>OBJECTIF</p>
        <p style='color:#00d4ff; font-family:Share Tech Mono,monospace; font-size:22px; margin:0;'>
            Détection IDS</p>
        <p style='color:#4a7a9b; font-size:12px; margin:4px 0 0 0;'>
            Classification d'attaques réseau</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Architecture du Projet")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Scripts Python du projet :**
        - `preprocessing.py` — Chargement & encodage des données
        - `eda.py` — Analyse exploratoire (EDA)
        - `compare_models.py` — Comparaison des 5 algorithmes
        - `feature_importance.py` — Importance des variables (RF)
        - `save_models.py` — Entraînement & sauvegarde du modèle
        - `test_model.py` — Test & évaluation sur données de test
        """)
    with col2:
        st.markdown("""
        **Fonctionnalités de cette interface :**
        - 📊 Visualisation et exploration du dataset KDD
        - ⚖️ Comparaison des métriques entre tous les modèles
        - 🌲 Analyse des features les plus importantes
        - 🔬 Évaluation détaillée sur le jeu de test
        - 💾 Sauvegarde et chargement du modèle entraîné
        """)

    st.markdown("---")
    st.subheader("🗺️ Guide d'utilisation")
    steps = [
        ("1️⃣", "EDA & Exploration", "Chargez le dataset et explorez la distribution des données"),
        ("2️⃣", "Comparaison Modèles", "Entraînez et comparez les 5 algorithmes ML"),
        ("3️⃣", "Feature Importance", "Identifiez les variables les plus pertinentes avec Random Forest"),
        ("4️⃣", "Test du Modèle", "Évaluez le modèle sauvegardé sur le jeu de test"),
        ("5️⃣", "Sauvegarde", "Sauvegardez le meilleur modèle en fichier .pkl"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class='stat-card' style='display:flex; align-items:center; gap:16px; padding:12px 20px;'>
            <span style='font-size:24px;'>{icon}</span>
            <div>
                <p style='color:#4fc3f7; font-family:Rajdhani,sans-serif; font-weight:700;
                          font-size:15px; margin:0;'>{title}</p>
                <p style='color:#8ab4d0; font-size:13px; margin:0;'>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 1 — EDA
# ─────────────────────────────────────────────
elif "EDA" in page:
    st.title("📊 Exploration & Analyse des Données")

    if not os.path.exists(train_path):
        st.error(f"❌ Fichier introuvable : `{train_path}`. Vérifiez le chemin dans la barre latérale.")
        st.stop()

    df = load_data(train_path)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Aperçu", "📈 Statistiques", "🏷️ Distribution Labels", "🔥 Corrélations"
    ])

    with tab1:
        st.subheader("Premières lignes du dataset")
        n = st.slider("Nombre de lignes à afficher", 5, 50, 10)
        st.dataframe(df.head(n), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes", f"{df.shape[0]:,}")
        c2.metric("Colonnes", f"{df.shape[1]}")
        c3.metric("Valeurs nulles", f"{df.isnull().sum().sum():,}")
        c4.metric("Doublons", f"{df.duplicated().sum():,}")

        st.subheader("Types des colonnes")
        dtype_df = pd.DataFrame({
            "Colonne": df.dtypes.index,
            "Type": df.dtypes.values.astype(str),
            "Nulls": df.isnull().sum().values,
            "Uniques": df.nunique().values
        })
        st.dataframe(dtype_df, use_container_width=True)

    with tab2:
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe().T.style.format("{:.3f}"), use_container_width=True)

        st.subheader("Distribution d'une feature")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected = st.selectbox("Sélectionner une colonne numérique", num_cols)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df[selected].dropna(), bins=40, color=ACCENT, alpha=0.8, edgecolor=GRID_COLOR)
        axes[0].set_title(f"Histogramme — {selected}")
        axes[0].set_xlabel(selected)
        axes[0].set_ylabel("Fréquence")

        axes[1].boxplot(df[selected].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#003d6b", color=ACCENT),
                        medianprops=dict(color="#ff6e00", linewidth=2),
                        whiskerprops=dict(color=ACCENT),
                        capprops=dict(color=ACCENT),
                        flierprops=dict(markerfacecolor=ACCENT, markersize=3))
        axes[1].set_title(f"Boxplot — {selected}")
        axes[1].set_ylabel(selected)

        apply_dark_theme(fig)
        st.pyplot(fig)
        plt.close()

    with tab3:
        if 'labels' not in df.columns:
            st.warning("Colonne 'labels' introuvable.")
        else:
            st.subheader("Distribution des classes / attaques")
            vc = df['labels'].value_counts()

            c1, c2 = st.columns([2, 1])
            with c1:
                fig, ax = plt.subplots(figsize=(12, 5))
                bars = ax.bar(range(len(vc)), vc.values,
                              color=[ACCENT if i == 0 else "#1a6a9a" for i in range(len(vc))],
                              edgecolor=GRID_COLOR, linewidth=0.5)
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels(vc.index, rotation=45, ha='right', fontsize=9)
                ax.set_title("Distribution des classes d'attaques")
                ax.set_ylabel("Nombre d'échantillons")
                apply_dark_theme(fig)
                st.pyplot(fig)
                plt.close()

            with c2:
                st.markdown("**Top 10 classes**")
                top_df = pd.DataFrame({
                    "Label": vc.index[:10],
                    "Count": vc.values[:10],
                    "%": (vc.values[:10] / vc.sum() * 100).round(2)
                })
                st.dataframe(top_df, use_container_width=True, hide_index=True)

            fig2, ax2 = plt.subplots(figsize=(7, 7))
            top_n = vc.head(8)
            wedge_props = {'linewidth': 2, 'edgecolor': BG_DARK}
            pie_colors = [f"#{hex(int(0x004080 + i * 0x001800))[2:]}ff" for i in range(len(top_n))]
            pie_colors = ["#00d4ff","#0090d4","#005a9e","#003d6b","#00bcd4",
                          "#0288d1","#01579b","#006064"][:len(top_n)]
            ax2.pie(top_n.values, labels=top_n.index, autopct='%1.1f%%',
                    colors=pie_colors, wedgeprops=wedge_props,
                    textprops={'color': TEXT_COLOR, 'fontsize': 9})
            ax2.set_title("Répartition des Top 8 classes", color=ACCENT, fontsize=13)
            fig2.patch.set_facecolor(BG_DARK)
            st.pyplot(fig2)
            plt.close()

    with tab4:
        st.subheader("Matrice de corrélation")
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] > 20:
            st.info("Affichage limité aux 20 premières colonnes numériques.")
            num_df = num_df.iloc[:, :20]

        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    ax=ax, annot=False, linewidths=0.3,
                    linecolor=GRID_COLOR, cbar_kws={"shrink": 0.7})
        ax.set_title("Corrélations entre features", color=ACCENT, fontsize=13)
        apply_dark_theme(fig)
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────
# PAGE 2 — COMPARAISON MODÈLES
# ─────────────────────────────────────────────
elif "Comparaison" in page:
    st.title("⚖️ Comparaison des Algorithmes ML")

    if not os.path.exists(train_path):
        st.error(f"❌ Fichier introuvable : `{train_path}`")
        st.stop()

    df = load_data(train_path)
    X, y = preprocess(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_models = st.multiselect(
            "Sélectionner les modèles à comparer",
            list(MODELS.keys()),
            default=list(MODELS.keys())
        )
    with col2:
        test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100

    if st.button("🚀 Lancer la Comparaison"):
        if not selected_models:
            st.warning("Veuillez sélectionner au moins un modèle.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            results = []
            prog = st.progress(0)
            status = st.empty()

            for i, name in enumerate(selected_models):
                status.markdown(f"⚙️ Entraînement : **{name}**…")
                model = MODELS[name]
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                t1 = time.time()
                y_pred = model.predict(X_test)
                pred_time = time.time() - t1

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                results.append({
                    "Modèle": name,
                    "Accuracy": round(acc, 4),
                    "Précision": round(prec, 4),
                    "Rappel": round(rec, 4),
                    "F1-Score": round(f1, 4),
                    "Temps train (s)": round(train_time, 2),
                    "Temps préd (s)": round(pred_time, 4),
                })
                prog.progress((i + 1) / len(selected_models))

            status.markdown("✅ Comparaison terminée !")
            st.session_state["compare_results"] = results
            st.session_state["compare_test"] = (X_test, y_test, selected_models)

    if "compare_results" in st.session_state:
        results = st.session_state["compare_results"]
        res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)

        st.subheader("🏆 Tableau des Résultats")
        best = res_df.iloc[0]
        st.success(f"🥇 Meilleur modèle : **{best['Modèle']}** — Accuracy: **{best['Accuracy']:.2%}** | F1: **{best['F1-Score']:.4f}**")
        st.dataframe(res_df.reset_index(drop=True), use_container_width=True)

        # ── Bar charts comparatifs
        st.subheader("📊 Visualisations Comparatives")
        metrics = ["Accuracy", "Précision", "Rappel", "F1-Score"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [r[metric] for r in results]
            names = [r["Modèle"] for r in results]
            colors = [COLORS.get(n, ACCENT) for n in names]
            bars = ax.barh(names, values, color=colors, edgecolor=GRID_COLOR,
                           linewidth=0.5, height=0.6)
            for bar, val in zip(bars, values):
                ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va='center', color=TEXT_COLOR, fontsize=9,
                        fontfamily='monospace')
            ax.set_xlim(0, min(1.12, max(values) * 1.15))
            ax.set_title(metric)
            ax.set_xlabel("Score")

        apply_dark_theme(fig)
        fig.tight_layout(pad=3)
        st.pyplot(fig)
        plt.close()

        # ── Temps d'entraînement
        st.subheader("⏱️ Temps d'Entraînement")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        names = [r["Modèle"] for r in results]
        times = [r["Temps train (s)"] for r in results]
        colors = [COLORS.get(n, ACCENT) for n in names]
        ax2.bar(names, times, color=colors, edgecolor=GRID_COLOR, linewidth=0.5)
        for i, (name, t) in enumerate(zip(names, times)):
            ax2.text(i, t + 0.01, f"{t:.2f}s", ha='center', color=TEXT_COLOR,
                     fontsize=9, fontfamily='monospace')
        ax2.set_title("Temps d'entraînement par modèle")
        ax2.set_ylabel("Secondes")
        apply_dark_theme(fig2)
        st.pyplot(fig2)
        plt.close()

        # ── Radar chart
        st.subheader("🕸️ Radar — Profil des Modèles")
        metrics_radar = ["Accuracy", "Précision", "Rappel", "F1-Score"]
        N = len(metrics_radar)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig3.patch.set_facecolor(BG_DARK)
        ax3.set_facecolor(BG_CARD)
        ax3.spines['polar'].set_color(GRID_COLOR)
        ax3.grid(color=GRID_COLOR, linewidth=0.5)

        palette = ["#00d4ff", "#4fc3f7", "#81d4fa", "#4dd0e1", "#26c6da"]
        for i, r in enumerate(results):
            vals = [r[m] for m in metrics_radar]
            vals += vals[:1]
            color = palette[i % len(palette)]
            ax3.plot(angles, vals, linewidth=2, linestyle='solid', color=color, label=r["Modèle"])
            ax3.fill(angles, vals, color=color, alpha=0.1)

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(metrics_radar, color=TEXT_COLOR, fontsize=11)
        ax3.set_ylim(0, 1)
        ax3.tick_params(colors=TEXT_COLOR)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                   facecolor=BG_CARD, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        ax3.set_title("Profil comparatif des modèles", color=ACCENT, pad=20, fontsize=13)
        st.pyplot(fig3)
        plt.close()


# ─────────────────────────────────────────────
# PAGE 3 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
elif "Feature" in page:
    st.title("🌲 Feature Importance — Random Forest")

    if not os.path.exists(train_path):
        st.error(f"❌ Fichier introuvable : `{train_path}`")
        st.stop()

    df = load_data(train_path)
    X, y = preprocess(df)

    top_n = st.slider("Nombre de features à afficher", 5, min(41, X.shape[1]), 20)

    if st.button("🌲 Calculer l'Importance des Features"):
        with st.spinner("Entraînement Random Forest…"):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        st.session_state["fi_df"] = fi_df

    if "fi_df" in st.session_state:
        fi_df = st.session_state["fi_df"]

        c1, c2 = st.columns([2, 1])
        with c2:
            st.subheader("Top Features")
            st.dataframe(fi_df.head(top_n)[["Feature", "Importance"]].style.format(
                {"Importance": "{:.4f}"}
            ), use_container_width=True, hide_index=True)

        with c1:
            st.subheader("Graphique — Features les plus importantes")
            top = fi_df.head(top_n)
            fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))

            palette = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))[::-1]
            bars = ax.barh(top["Feature"][::-1], top["Importance"][::-1],
                           color=palette, edgecolor=GRID_COLOR, linewidth=0.4)

            for bar, val in zip(bars, top["Importance"][::-1]):
                ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va='center', color=TEXT_COLOR,
                        fontsize=8, fontfamily='monospace')

            ax.set_title(f"Top {top_n} Features — Random Forest", fontsize=13)
            ax.set_xlabel("Importance")
            apply_dark_theme(fig)
            st.pyplot(fig)
            plt.close()

        st.subheader("📊 Cumulative Importance")
        fi_df["Cumulative"] = fi_df["Importance"].cumsum()
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(range(1, len(fi_df) + 1), fi_df["Cumulative"],
                 color=ACCENT, linewidth=2)
        ax2.axhline(y=0.9, color="#ff6e00", linestyle='--', linewidth=1, alpha=0.7,
                    label="90% importance")
        ax2.axhline(y=0.95, color="#ff3300", linestyle='--', linewidth=1, alpha=0.7,
                    label="95% importance")
        idx_90 = (fi_df["Cumulative"] >= 0.90).idxmax() + 1
        idx_95 = (fi_df["Cumulative"] >= 0.95).idxmax() + 1
        ax2.axvline(x=idx_90, color="#ff6e00", linestyle=':', linewidth=1, alpha=0.5)
        ax2.axvline(x=idx_95, color="#ff3300", linestyle=':', linewidth=1, alpha=0.5)
        ax2.fill_between(range(1, len(fi_df) + 1), fi_df["Cumulative"],
                         alpha=0.15, color=ACCENT)
        ax2.set_title("Importance cumulée des features")
        ax2.set_xlabel("Nombre de features (triées par importance)")
        ax2.set_ylabel("Importance cumulée")
        ax2.legend(facecolor=BG_CARD, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        apply_dark_theme(fig2)
        st.pyplot(fig2)
        plt.close()

        st.info(f"ℹ️ **{idx_90} features** suffisent pour couvrir 90% de l'importance  |  "
                f"**{idx_95} features** pour 95%")


# ─────────────────────────────────────────────
# PAGE 4 — TEST DU MODÈLE
# ─────────────────────────────────────────────
elif "Test" in page:
    st.title("🔬 Test & Évaluation du Modèle")

    if not os.path.exists(model_path):
        st.warning(f"⚠️ Modèle `{model_path}` non trouvé. Entraînez et sauvegardez d'abord le modèle.")

        if not os.path.exists(train_path):
            st.error(f"❌ Dataset d'entraînement introuvable : `{train_path}`")
            st.stop()

        if st.button("⚡ Entraîner & Tester directement"):
            df_train = load_data(train_path)
            X_all, y_all = preprocess(df_train)
            X_train, X_test_s, y_train, y_test_s = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42
            )
            with st.spinner("Entraînement Random Forest…"):
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
            st.session_state["inline_model"] = model
            st.session_state["inline_test"] = (X_test_s, y_test_s)
            st.success("✅ Modèle entraîné sur le split interne (80/20).")

    if "inline_model" in st.session_state:
        model = st.session_state["inline_model"]
        X_test, y_test = st.session_state["inline_test"]
    elif os.path.exists(model_path):
        model = joblib.load(model_path)
        if not os.path.exists(test_path):
            st.error(f"❌ Fichier test introuvable : `{test_path}`")
            st.stop()
        df_test = load_data(test_path)
        X_test, y_test = preprocess(df_test)
    else:
        st.stop()

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Accuracy", f"{acc:.2%}")
    c2.metric("📍 Précision", f"{prec:.4f}")
    c3.metric("🔁 Rappel", f"{rec:.4f}")
    c4.metric("📐 F1-Score", f"{f1:.4f}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🔲 Matrice de Confusion", "📄 Rapport Détaillé", "📊 Distribution Prédictions"])

    with tab1:
        st.subheader("Matrice de Confusion")
        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        if len(labels) > 20:
            st.info(f"⚠️ {len(labels)} classes détectées — affichage normalisé.")
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
            fig, ax = plt.subplots(figsize=(14, 11))
            sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax,
                        linewidths=0.2, linecolor=GRID_COLOR)
        else:
            fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 1)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax,
                        linewidths=0.5, linecolor=GRID_COLOR)

        ax.set_xlabel("Prédit", fontsize=11)
        ax.set_ylabel("Réel", fontsize=11)
        ax.set_title("Matrice de Confusion", fontsize=13)
        apply_dark_theme(fig)
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Rapport de Classification")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.style.format("{:.4f}").background_gradient(
            cmap="Blues", subset=["precision", "recall", "f1-score"]
        ), use_container_width=True)

    with tab3:
        st.subheader("Distribution Réel vs Prédit")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        vc_true = pd.Series(y_test).value_counts().head(15)
        axes[0].bar(range(len(vc_true)), vc_true.values, color=ACCENT,
                    edgecolor=GRID_COLOR, linewidth=0.4)
        axes[0].set_xticks(range(len(vc_true)))
        axes[0].set_xticklabels(vc_true.index, rotation=45, ha='right', fontsize=8)
        axes[0].set_title("Distribution réelle (y_test)")
        axes[0].set_ylabel("Count")

        vc_pred = pd.Series(y_pred).value_counts().head(15)
        axes[1].bar(range(len(vc_pred)), vc_pred.values, color="#4fc3f7",
                    edgecolor=GRID_COLOR, linewidth=0.4)
        axes[1].set_xticks(range(len(vc_pred)))
        axes[1].set_xticklabels(vc_pred.index, rotation=45, ha='right', fontsize=8)
        axes[1].set_title("Distribution prédite (y_pred)")
        axes[1].set_ylabel("Count")

        apply_dark_theme(fig)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────
# PAGE 5 — SAUVEGARDE
# ─────────────────────────────────────────────
elif "Sauvegarde" in page:
    st.title("💾 Sauvegarde du Modèle")

    if not os.path.exists(train_path):
        st.error(f"❌ Dataset introuvable : `{train_path}`")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Algorithme à sauvegarder", list(MODELS.keys()),
                                    index=1)
    with col2:
        save_name = st.text_input("Nom du fichier", value=model_path)

    if st.button("💾 Entraîner & Sauvegarder"):
        df = load_data(train_path)
        X, y = preprocess(df)
        model = MODELS[model_choice]

        prog = st.progress(0)
        with st.spinner(f"Entraînement de {model_choice}…"):
            t0 = time.time()
            model.fit(X, y)
            elapsed = time.time() - t0
        prog.progress(70)

        joblib.dump(model, save_name)
        prog.progress(100)

        file_size = os.path.getsize(save_name) / 1024

        st.success(f"✅ Modèle **{model_choice}** sauvegardé dans `{save_name}`")

        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithme", model_choice)
        c2.metric("Temps d'entraînement", f"{elapsed:.2f}s")
        c3.metric("Taille du fichier", f"{file_size:.1f} KB")

        st.markdown("---")
        st.subheader("📋 Résumé du Modèle Sauvegardé")
        st.markdown(f"""
        <div class='stat-card'>
            <p style='color:#4fc3f7; font-family:Rajdhani,sans-serif; font-weight:700; margin:0;'>
                INFORMATIONS DU MODÈLE
            </p>
            <hr style='border-color:#1a3a5c;'/>
            <p style='color:#8ab4d0; margin:4px 0; font-family:Share Tech Mono,monospace; font-size:13px;'>
                Algorithme : <span style='color:#00d4ff;'>{model_choice}</span>
            </p>
            <p style='color:#8ab4d0; margin:4px 0; font-family:Share Tech Mono,monospace; font-size:13px;'>
                Dataset d'entraînement : <span style='color:#00d4ff;'>{train_path}</span>
            </p>
            <p style='color:#8ab4d0; margin:4px 0; font-family:Share Tech Mono,monospace; font-size:13px;'>
                Features : <span style='color:#00d4ff;'>{X.shape[1]}</span>
            </p>
            <p style='color:#8ab4d0; margin:4px 0; font-family:Share Tech Mono,monospace; font-size:13px;'>
                Échantillons d'entraînement : <span style='color:#00d4ff;'>{X.shape[0]:,}</span>
            </p>
            <p style='color:#8ab4d0; margin:4px 0; font-family:Share Tech Mono,monospace; font-size:13px;'>
                Classes détectées : <span style='color:#00d4ff;'>{y.nunique()}</span>
            </p>
            <p style='color:#8ab4d0; margin:4px 0; font-family:Share Tech Mono,monospace; font-size:13px;'>
                Fichier sauvegardé : <span style='color:#00d4ff;'>{save_name}</span>
                (<span style='color:#00d4ff;'>{file_size:.1f} KB</span>)
            </p>
        </div>
        """, unsafe_allow_html=True)

    if os.path.exists(save_name):
        st.markdown("---")
        st.subheader("🔍 Vérifier le Modèle Sauvegardé")
        if st.button("📂 Charger et vérifier le modèle"):
            loaded = joblib.load(save_name)
            st.success(f"✅ Modèle chargé avec succès : `{type(loaded).__name__}`")
            df = load_data(train_path)
            X, y = preprocess(df)
            _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=99)
            y_p = loaded.predict(X_val)
            vacc = accuracy_score(y_val, y_p)
            st.metric("Accuracy de validation (20%)", f"{vacc:.2%}")