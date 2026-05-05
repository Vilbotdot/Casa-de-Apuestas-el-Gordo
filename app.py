import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Casa de Apuestas El Gordo", page_icon="⚽", layout="wide")

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
    
    # Entrenamiento ML (Sklearn)
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
st.title("CASA DE APUESTAS EL GORDO")
st.write("---")

liga_sel = st.selectbox("Selecciona el torneo:", list(LIGAS.keys()))

try:
    raw_matches, df, ml_model, scaler = entrenar_y_cargar(LIGAS[liga_sel])
    equipos_lista = sorted(df['Squad'].unique())
    
    col_input1, col_input2 = st.columns(2)
    with col_input1: loc = st.selectbox("Local", equipos_lista)
    with col_input2: vis = st.selectbox("Visita", [e for e in equipos_lista if e != loc])
    
    if st.button("ANALIZAR PARTIDO", use_container_width=True):
        # Cálculos Base
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
        
        # Matriz Poisson
        max_g = 10
        matriz = np.outer(poisson.pmf(range(max_g), l_l), poisson.pmf(range(max_g), l_v))
        matriz /= matriz.sum()
        
        # Probabilidades Poisson
        p_loc_poi = np.sum(np.tril(matriz, -1)) * 100
        p_emp_poi = np.sum(np.diag(matriz)) * 100
        p_vis_poi = np.sum(np.triu(matriz, 1)) * 100
        
        # Probabilidades Machine Learning (Sklearn)
        ventaja_l = (d_l['Elo'] + 100) / max(d_v['Elo'], 1)
        X_new = scaler.transform([[1 / ventaja_l, ventaja_l]])
        probs_ml = ml_model.predict_proba(X_new)[0]
        clases = list(ml_model.classes_)
        p_emp_ml = (probs_ml[clases.index(0)] * 100) if 0 in clases else 0
        p_loc_ml = (probs_ml[clases.index(1)] * 100) if 1 in clases else 0
        p_vis_ml = (probs_ml[clases.index(2)] * 100) if 2 in clases else 0
        
        # Goles Over/Under
        goles = np.add.outer(range(max_g), range(max_g))
        def calc_ou(n): return np.sum(matriz[goles > n]) * 100
        
        # --- MOSTRAR RESULTADOS ---
        st.header(f"{loc} vs {vis}")
        
        # 1. PROBABILIDADES 1X2 (ML vs POISSON)
        st.subheader("--- PROBABILIDADES 1X2 ---")
        col_ml, col_poi = st.columns(2)
        with col_ml:
            st.write("**SKLEARN ML (IA)**")
            st.write(f"Local: {p_loc_ml:.1f}% | Empate: {p_emp_ml:.1f}% | Visita: {p_vis_ml:.1f}%")
        with col_poi:
            st.write("**POISSON TRADICIONAL**")
            st.write(f"Local: {p_loc_poi:.1f}% | Empate: {p_emp_poi:.1f}% | Visita: {p_vis_poi:.1f}%")

        # 2. GOLES ESPERADOS
        st.subheader("--- GOLES ESPERADOS (xG) ---")
        st.info(f"{loc}: {l_l:.2f} | {vis}: {l_v:.2f}")

        # 3. OVER / UNDER
        st.subheader("--- OVER / UNDER ---")
        o_cols = st.columns(5)
        for i, val in enumerate([0.5, 1.5, 2.5, 3.5, 4.5]):
            p_over = calc_ou(val)
            o_cols[i].metric(f"+{val}", f"{p_over:.1f}%")
            o_cols[i].write(f"-{val}: {100-p_over:.1f}%")
        
        btts = np.sum(matriz[1:, 1:]) * 100
        st.write(f"**Ambos Equipos Anotan (BTTS):** {btts:.1f}%")

        # 4. TOP 5 MARCADORES
        st.subheader("--- TOP 5 MARCADORES ---")
        m_list = []
        for i in range(max_g):
            for j in range(max_g): m_list.append((matriz[i,j], i, j))
        m_list.sort(reverse=True)
        
        for i in range(5):
            p_m, gl, gv = m_list[i]
            st.write(f"{i+1}. {loc} **{gl} - {gv}** {vis} ({p_m*100:.1f}%)")

        # 5. RECOMENDACIONES FINALES
        st.subheader("--- RECOMENDACIONES CASA DE APUESTAS EL GORDO ---")
        favorito = None
        if p_loc_ml >= 45:
            st.success("MoneyLine: Gana Local (Confianza Sklearn)")
            favorito = "Local"
        elif p_vis_ml >= 45:
            st.success("MoneyLine: Gana Visita (Confianza Sklearn)")
            favorito = "Visita"
        else:
            favorito = "Local_DC" if p_loc_ml > p_vis_ml else "Visita_DC"
            st.warning(f"Doble Oportunidad: {'Local o Empate' if favorito == 'Local_DC' else 'Visita o Empate'}")

        if calc_ou(2.5) >= 55: st.info("Goles: Over 2.5 tiene valor")
        if btts >= 58: st.info("Ambos Equipos Anotan: SI")

        # Combinadas
        st.subheader("--- COMBINADAS DE ALTO VALOR ---")
        u35 = 100 - calc_ou(3.5)
        o15 = calc_ou(1.5)
        
        if favorito == "Local" and u35 >= 75: st.write("Combo: Local Gana + Under 3.5 Goles")
        if favorito == "Visita" and u35 >= 75: st.write("Combo: Visita Gana + Under 3.5 Goles")
        if favorito in ["Local", "Local_DC"] and o15 >= 75: st.write("Combo Seguro: Local o Empate + Over 1.5 Goles")
        if favorito in ["Visita", "Visita_DC"] and o15 >= 75: st.write("Combo Seguro: Visita o Empate + Over 1.5 Goles")

except Exception as e:
    st.error(f"Error cargando la liga: {e}")
