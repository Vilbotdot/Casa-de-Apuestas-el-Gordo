import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="EasyBet", page_icon="⚽")

# Cacheamos los datos para no descargar el CSV cada vez que tocas un botón
@st.cache_data
def cargar_datos_online():
    url = "https://www.football-data.co.uk/new/MEX.csv"
    raw = pd.read_csv(url)
    raw = raw[raw['Season'] == '2025/2026']
    raw = raw.dropna(subset=['HG', 'AG'])
    
    equipos = {}
    todos = pd.concat([raw['Home'], raw['Away']]).unique()
    
    for equipo in todos:
        partidos = raw[(raw['Home'] == equipo) | (raw['Away'] == equipo)].tail(12)
        GF, GA, MP, PTS = 0, 0, 0, 0
        
        for _, row in partidos.iterrows():
            if row['Home'] == equipo:
                gf, ga = row['HG'], row['AG']
            else:
                gf, ga = row['AG'], row['HG']
                
            GF += gf
            GA += ga
            MP += 1
            if gf > ga: PTS += 3
            elif gf == ga: PTS += 1
                
        elo = 1500 + (GF - GA) * 8 + PTS * 2
        equipos[equipo] = {"Squad": equipo, "GF": GF, "GA": GA, "MP": MP, "Elo": elo}
        
    df = pd.DataFrame(list(equipos.values()))
    return raw, df

def obtener_sede_stats(raw_matches, equipo):
    partidos_local = raw_matches[raw_matches['Home'] == equipo].tail(5)
    partidos_visita = raw_matches[raw_matches['Away'] == equipo].tail(5)
    return {
        "local_gf": partidos_local['HG'].sum(),
        "local_ga": partidos_local['AG'].sum(),
        "local_mp": max(len(partidos_local), 1),
        "visita_gf": partidos_visita['AG'].sum(),
        "visita_ga": partidos_visita['HG'].sum(),
        "visita_mp": max(len(partidos_visita), 1)
    }

def generar_recomendacion(prob_local, prob_empate, prob_visita, over05, over15, over25, over35, over45, under25, under35, btts):
    recomendaciones = []
    if prob_local >= 47: recomendaciones.append("Money Line recomendado: **Gana Local**")
    elif prob_visita >= 47: recomendaciones.append("Money Line recomendado: **Gana Visita**")
    else:
        if prob_local > prob_visita: recomendaciones.append("Doble oportunidad: **Local o Empate**")
        elif prob_visita > prob_local: recomendaciones.append("Doble oportunidad: **Visita o Empate**")
        else: recomendaciones.append("Partido muy equilibrado en 1X2")
        
    if prob_empate >= 30: recomendaciones.append("El **Empate** tiene valor estadístico")
    if over15 >= 72: recomendaciones.append("Pick fuerte: **Over 1.5 goles**")
    if over25 >= 54: recomendaciones.append("Pick agresivo: **Over 2.5 goles**")
    if under25 >= 56: recomendaciones.append("Pick defensivo: **Under 2.5 goles**")
    if under35 >= 72: recomendaciones.append("Pick conservador: **Under 3.5 goles**")
    if btts >= 58: recomendaciones.append("**Ambos equipos anotan** tiene valor")
    
    if len(recomendaciones) <= 1: recomendaciones.append("Sin valor estadístico claro")
    return recomendaciones

# --- INTERFAZ WEB ---
st.title("⚽ EasyBet - Predictor Liga MX")

try:
    raw_matches, df = cargar_datos_online()
    equipos_lista = sorted(df['Squad'].unique())
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1: loc = st.selectbox("Local", equipos_lista)
    with col2: st.markdown("<h3 style='text-align: center; margin-top: 25px;'>VS</h3>", unsafe_allow_html=True)
    with col3: vis = st.selectbox("Visita", reversed(equipos_lista))
    
    if st.button("Analizar Partido", use_container_width=True):
        if loc == vis:
            st.error("Selecciona equipos diferentes")
        else:
            # --- LÓGICA MATEMÁTICA ---
            d_l = df[df['Squad'] == loc].iloc[0]
            d_v = df[df['Squad'] == vis].iloc[0]
            sede_l = obtener_sede_stats(raw_matches, loc)
            sede_v = obtener_sede_stats(raw_matches, vis)
            
            ataque_local = ((d_l['GF']/d_l['MP']) * 0.6) + ((sede_l['local_gf']/sede_l['local_mp']) * 0.4)
            defensa_local = ((d_l['GA']/d_l['MP']) * 0.6) + ((sede_l['local_ga']/sede_l['local_mp']) * 0.4)
            ataque_visita = ((d_v['GF']/d_v['MP']) * 0.6) + ((sede_v['visita_gf']/sede_v['visita_mp']) * 0.4)
            defensa_visita = ((d_v['GA']/d_v['MP']) * 0.6) + ((sede_v['visita_ga']/sede_v['visita_mp']) * 0.4)
            
            l_l = max(((ataque_local + defensa_visita) / 2) * 1.18, 0.1)
            l_v = max(((ataque_visita + defensa_local) / 2), 0.1)
            
            elo_l, elo_v = d_l['Elo'], d_v['Elo']
            ajuste = (elo_l - elo_v) / 1800
            l_l *= (1 + ajuste)
            l_v *= (1 - ajuste)
            
            prob_local_elo = (1 / (1 + 10 ** ((elo_v - elo_l) / 400))) * 100
            prob_visita_elo = 100 - prob_local_elo
            
            max_goles = 10
            matriz = np.outer(poisson.pmf(range(max_goles), l_l), poisson.pmf(range(max_goles), l_v))
            matriz = matriz / matriz.sum()
            
            prob_local = np.sum(np.tril(matriz, -1)) * 100
            prob_empate = np.sum(np.diag(matriz)) * 100
            prob_visita = np.sum(np.triu(matriz, 1)) * 100
            
            goles = np.add.outer(range(max_goles), range(max_goles))
            def calc_over(n): return np.sum(matriz[goles > n]) * 100
            
            over05, over15, over25, over35, over45 = calc_over(0.5), calc_over(1.5), calc_over(2.5), calc_over(3.5), calc_over(4.5)
            under05, under15, under25, under35, under45 = 100-over05, 100-over15, 100-over25, 100-over35, 100-over45
            btts = np.sum(matriz[1:, 1:]) * 100
            
            marcadores = []
            for i in range(max_goles):
                for j in range(max_goles):
                    marcadores.append((matriz[i, j] * 100, i, j))
            marcadores.sort(reverse=True)
            
            recoms = generar_recomendacion(prob_local, prob_empate, prob_visita, over05, over15, over25, over35, over45, under25, under35, btts)

            # --- RENDERIZADO DE RESULTADOS ---
            st.success("Análisis completado exitosamente")
            
            st.subheader("Recomendación EasyBet 💡")
            for r in recoms:
                st.info(r)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Prob. {loc}", f"{prob_local:.1f}%")
            c2.metric("Empate", f"{prob_empate:.1f}%")
            c3.metric(f"Prob. {vis}", f"{prob_visita:.1f}%")
            
            with st.expander("Ver estadísticas detalladas"):
                st.write(f"**Goles Esperados:** {loc} ({l_l:.2f}) vs {vis} ({l_v:.2f})")
                st.write(f"**Elo Dinámico:** {loc} ({elo_l:.0f}) vs {vis} ({elo_v:.0f})")
                st.write(f"**Jerarquía:** {loc} ({prob_local_elo:.1f}%) vs {vis} ({prob_visita_elo:.1f}%)")
                st.write(f"**Ambos Anotan (BTTS):** {btts:.1f}%")
                
                col_over, col_under = st.columns(2)
                with col_over:
                    st.write("**Overs:**")
                    st.write(f"+1.5: {over15:.1f}% | +2.5: {over25:.1f}%")
                with col_under:
                    st.write("**Unders:**")
                    st.write(f"-2.5: {under25:.1f}% | -3.5: {under35:.1f}%")
                    
                st.write("**Top 5 Marcadores Probables:**")
                for i in range(5):
                    p, gl, gv = marcadores[i]
                    st.write(f"{i+1}. {loc} **{gl} - {gv}** {vis} ({p:.1f}%)")

except Exception as e:
    st.error(f"Error cargando datos: {e}")