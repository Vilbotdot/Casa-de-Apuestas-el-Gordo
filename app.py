import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Casa de Apuestas El Gordo", page_icon="⚽")

# --- CONFIGURACIÓN Y DATOS ---
LIGAS = {
    "Mexico (Liga MX)": "https://www.football-data.co.uk/new/MEX.csv",
    "USA (MLS)": "https://www.football-data.co.uk/new/USA.csv",
    "Brasil (Serie A)": "https://www.football-data.co.uk/new/BRA.csv",
    "Argentina (Liga Prof)": "https://www.football-data.co.uk/new/ARG.csv",
    "Inglaterra (Premier League)": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "Espana (LaLiga)": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "Italia (Serie A)": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "Alemania (Bundesliga)": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "Francia (Ligue 1)": "https://www.football-data.co.uk/mmz4281/2526/F1.csv"
}

@st.cache_resource
def entrenar_y_cargar(url):
    raw = pd.read_csv(url)
    rename_map = {'HomeTeam': 'Home', 'AwayTeam': 'Away', 'FTHG': 'HG', 'FTAG': 'AG'}
    raw = raw.rename(columns=rename_map)
    
    if 'Season' in raw.columns:
        ultima_temporada = raw['Season'].dropna().iloc[-1]
        raw = raw[raw['Season'] == ultima_temporada]
    
    raw = raw.dropna(subset=['Home', 'Away', 'HG', 'AG'])
    
    equipos_stats = {}
    todos = pd.concat([raw['Home'], raw['Away']]).unique()
    
    for equipo in todos:
        partidos = raw[(raw['Home'] == equipo) | (raw['Away'] == equipo)].tail(12)
        GF, GA, MP, PTS = 0, 0, 0, 0
        for _, row in partidos.iterrows():
            if row['Home'] == equipo: gf, ga = row['HG'], row['AG']
            else: gf, ga = row['AG'], row['HG']
            GF += gf; GA += ga; MP += 1
            if gf > ga: PTS += 3
            elif gf == ga: PTS += 1
        
        elo = 1500 + (GF - GA) * 8 + PTS * 2
        equipos_stats[equipo] = {"Squad": equipo, "GF": GF, "GA": GA, "MP": MP, "Elo": elo}
    
    df_stats = pd.DataFrame(list(equipos_stats.values()))
    
    X_train, y_train = [], []
    for _, row in raw.iterrows():
        if row['HG'] > row['AG']: y = 1
        elif row['HG'] == row['AG']: y = 0
        else: y = 2
        elo_h = equipos_stats.get(row['Home'], {}).get('Elo', 1500)
        elo_a = equipos_stats.get(row['Away'], {}).get('Elo', 1500)
        ventaja = (elo_h + 100) / max(elo_a, 1)
        X_train.append([1 / ventaja, ventaja])
        y_train.append(y)
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_scaled, y_train)
    
    return raw, df_stats, model, scaler

def obtener_sede_stats(raw_matches, equipo):
    p_local = raw_matches[raw_matches['Home'] == equipo].tail(5)
    p_visita = raw_matches[raw_matches['Away'] == equipo].tail(5)
    return {
        "local_gf": p_local['HG'].sum(), "local_ga": p_local['AG'].sum(), "local_mp": max(len(p_local), 1),
        "visita_gf": p_visita['AG'].sum(), "visita_ga": p_visita['HG'].sum(), "visita_mp": max(len(p_visita), 1)
    }

# --- INTERFAZ ---
st.title("Casa de Apuestas El Gordo")
st.subheader("Analisis Global con Inteligencia Artificial")

liga_sel = st.selectbox("Selecciona el torneo:", list(LIGAS.keys()))

try:
    raw_matches, df, ml_model, scaler = entrenar_y_cargar(LIGAS[liga_sel])
    equipos_lista = sorted(df['Squad'].unique())
    
    col1, col2 = st.columns(2)
    with col1: loc = st.selectbox("Local", equipos_lista)
    with col2: vis = st.selectbox("Visita", [e for e in equipos_lista if e != loc])
    
    if st.button("Analizar Partido", use_container_width=True):
        d_l = df[df['Squad'] == loc].iloc[0]
        d_v = df[df['Squad'] == vis].iloc[0]
        sede_l = obtener_sede_stats(raw_matches, loc)
        sede_v = obtener_sede_stats(raw_matches, vis)
        
        at_l = ((d_l['GF']/d_l['MP']) * 0.6) + ((sede_l['local_gf']/sede_l['local_mp']) * 0.4)
        df_l = ((d_l['GA']/d_l['MP']) * 0.6) + ((sede_l['local_ga']/sede_l['local_mp']) * 0.4)
        at_v = ((d_v['GF']/d_v['MP']) * 0.6) + ((sede_v['visita_gf']/sede_v['visita_mp']) * 0.4)
        df_v = ((d_v['GA']/d_v['MP']) * 0.6) + ((sede_v['visita_ga']/sede_v['visita_mp']) * 0.4)
        
        l_l = max(((at_l + df_v) / 2) * 1.18, 0.1)
        l_v = max(((at_v + df_l) / 2), 0.1)
        ajuste = (d_l['Elo'] - d_v['Elo']) / 1800
        l_l *= (1 + ajuste); l_v *= (1 - ajuste)
        
        matriz = np.outer(poisson.pmf(range(10), l_l), poisson.pmf(range(10), l_v))
        matriz /= matriz.sum()
        
        ventaja_l = (d_l['Elo'] + 100) / max(d_v['Elo'], 1)
        X_new = scaler.transform([[1 / ventaja_l, ventaja_l]])
        probs_ml = ml_model.predict_proba(X_new)[0]
        clases = list(ml_model.classes_)
        p_emp_ml = (probs_ml[clases.index(0)] * 100) if 0 in clases else 0
        p_loc_ml = (probs_ml[clases.index(1)] * 100) if 1 in clases else 0
        p_vis_ml = (probs_ml[clases.index(2)] * 100) if 2 in clases else 0
        
        goles = np.add.outer(range(10), range(10))
        over25 = np.sum(matriz[goles > 2.5]) * 100
        btts = np.sum(matriz[1:, 1:]) * 100

        st.success("Analisis de IA Completado")
        
        st.write("### Recomendaciones Casa de Apuestas El Gordo")
        if p_loc_ml >= 45: st.info("MoneyLine: Gana Local (Confianza Sklearn)")
        elif p_vis_ml >= 45: st.info("MoneyLine: Gana Visita (Confianza Sklearn)")
        else: st.info("Doble Oportunidad sugerida por equilibrio")
        
        if over25 >= 55: st.info("Goles: Over 2.5 tiene valor")
        if btts >= 58: st.info("Ambos Equipos Anotan: SI")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Prob. {loc}", f"{p_loc_ml:.1f}%")
        c2.metric("Empate", f"{p_emp_ml:.1f}%")
        c3.metric(f"Prob. {vis}", f"{p_vis_ml:.1f}%")

        with st.expander("Ver detalle técnico"):
            st.write(f"Goles esperados: {loc} ({l_l:.2f}) - {vis} ({l_v:.2f})")
            st.write("--- Top 3 Marcadores ---")
            m_list = []
            for i in range(10):
                for j in range(10): m_list.append((matriz[i,j], i, j))
            m_list.sort(reverse=True)
            for i in range(3):
                p_m, gl, gv = m_list[i]
                st.write(f"{loc} **{gl} - {gv}** {vis} ({p_m*100:.1f}%)")

except Exception as e:
    st.error(f"Error cargando la liga: {e}")
