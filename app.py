import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from supabase import create_client, Client

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y SUPABASE
# ==========================================
st.set_page_config(page_title="El Gordo Picks", layout="centered", page_icon="🎲")

@st.cache_resource
def init_connection():
    # Lee de tu archivo .streamlit/secrets.toml
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase: Client = init_connection()
except Exception as e:
    st.error(f"Error de conexión a Supabase. Revisa tus secretos. Detalle: {e}")
    st.stop()

# ==========================================
# 2. CONTROL DE SESIONES
# ==========================================
if 'usuario_id' not in st.session_state:
    st.session_state.usuario_id = None
if 'model_data' not in st.session_state:
    st.session_state.model_data = None

# ==========================================
# 3. FUNCIONES CORE (MATEMÁTICAS)
# ==========================================
def obtener_sede_stats(raw_matches, equipo):
    partidos_local = raw_matches[raw_matches['Home'] == equipo].tail(5)
    partidos_visita = raw_matches[raw_matches['Away'] == equipo].tail(5)

    local_gf = partidos_local['HG'].sum(); local_ga = partidos_local['AG'].sum()
    local_mp = max(len(partidos_local), 1)

    visita_gf = partidos_visita['AG'].sum(); visita_ga = partidos_visita['HG'].sum()
    visita_mp = max(len(partidos_visita), 1)

    return {
        "local_gf": local_gf, "local_ga": local_ga, "local_mp": local_mp,
        "visita_gf": visita_gf, "visita_ga": visita_ga, "visita_mp": visita_mp
    }

def generar_recomendacion(prob_local_ml, prob_empate_ml, prob_visita_ml, over05, over15, over25, over35, over45, under25, under35, btts):
    recomendaciones = []
    favorito = None
    if prob_local_ml >= 45:
        recomendaciones.append("✅ **MoneyLine:** Gana Local (Confianza Sklearn)")
        favorito = "Local"
    elif prob_visita_ml >= 45:
        recomendaciones.append("✅ **MoneyLine:** Gana Visita (Confianza Sklearn)")
        favorito = "Visita"
    else:
        if prob_local_ml > prob_visita_ml:
            recomendaciones.append("⚠️ **Doble oportunidad:** Local o Empate")
            favorito = "Local_DC"
        else:
            recomendaciones.append("⚠️ **Doble oportunidad:** Visita o Empate")
            favorito = "Visita_DC"

    if over25 >= 55: recomendaciones.append("⚽ **Goles:** Over 2.5 tiene valor")
    elif under25 >= 58: recomendaciones.append("🛡️ **Goles:** Pick defensivo Under 2.5")
        
    if btts >= 58: recomendaciones.append("🔥 **Ambos Equipos Anotan:** SI")

    recomendaciones.append("\n**--- COMBINADAS DE ALTO VALOR ---**")
    if favorito == "Local" and under35 >= 75: recomendaciones.append("💸 **Combo:** Local Gana + Under 3.5 Goles")
    elif favorito == "Visita" and under35 >= 75: recomendaciones.append("💸 **Combo:** Visita Gana + Under 3.5 Goles")
        
    if favorito in ["Local", "Local_DC"] and over15 >= 75: recomendaciones.append("🔒 **Combo Seguro:** Local o Empate + Over 1.5 Goles")
    elif favorito in ["Visita", "Visita_DC"] and over15 >= 75: recomendaciones.append("🔒 **Combo Seguro:** Visita o Empate + Over 1.5 Goles")
        
    if favorito in ["Local_DC", "Visita_DC"] and btts >= 60: recomendaciones.append("💥 **Combo Goles:** Ambos Equipos Anotan + Over 2.5 Goles")

    return "\n".join(recomendaciones)

# ==========================================
# 4. PANTALLA DE LOGIN
# ==========================================
def pantalla_login():
    st.title("🔐 Casa de Apuestas El Gordo - Web VIP")
    
    try:
        st.image("image_6.jpg", width=100)
    except Exception:
        st.warning("Imagen de logo no encontrada.")
    
    t_login, t_registro = st.tabs(["Iniciar Sesión", "Crear Cuenta"])
    
    with t_login:
        email_login = st.text_input("Correo electrónico", key="log_e")
        pass_login = st.text_input("Contraseña", type="password", key="log_p")
        if st.button("Entrar", type="primary"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email_login, "password": pass_login})
                st.session_state.usuario_id = res.user.id
                st.rerun()
            except Exception as e:
                st.error(f"Credenciales incorrectas: {e}")

    with t_registro:
        email_reg = st.text_input("Nuevo Correo", key="reg_e")
        pass_reg = st.text_input("Contraseña (min 6)", type="password", key="reg_p")
        if st.button("Registrarse"):
            try:
                supabase.auth.sign_up({"email": email_reg, "password": pass_reg})
                st.success("¡Cuenta creada! Ya puedes iniciar sesión.")
            except Exception as e:
                st.error(f"Error al registrar: {e}")

# ==========================================
# 5. APLICACIÓN PRINCIPAL
# ==========================================
if st.session_state.usuario_id is None:
    pantalla_login()
else:
    st.sidebar.title("🎲 El Gordo Picks")
    try:
        st.sidebar.image("image_6.jpg", width=75)
    except Exception:
        pass
        
    st.sidebar.success("Sesión Activa")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.usuario_id = None
        st.session_state.model_data = None
        st.rerun()

    tab_calc, tab_hist = st.tabs(["📊 Calculadora de Picks", "📜 Mi Historial"])

    # ----------------------------------------
    # PESTAÑA 1: CALCULADORA Y PREDICCIÓN
    # ----------------------------------------
    with tab_calc:
        st.header("Análisis Global de Ligas")
        
        ligas = {
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

        liga_seleccionada = st.selectbox("Selecciona el torneo:", list(ligas.keys()))

        if st.button("📥 Importar Datos y Entrenar IA"):
            with st.spinner("Extrayendo estadísticas y entrenando modelo Sklearn..."):
                try:
                    url = ligas[liga_seleccionada]
                    raw = pd.read_csv(url)
                    rename_map = {'HomeTeam': 'Home', 'AwayTeam': 'Away', 'FTHG': 'HG', 'FTAG': 'AG'}
                    raw = raw.rename(columns=rename_map)

                    if 'Season' in raw.columns:
                        ultima_temporada = raw['Season'].dropna().iloc[-1]
                        raw = raw[raw['Season'] == ultima_temporada]

                    raw = raw.dropna(subset=['Home', 'Away', 'HG', 'AG'])
                    todos = pd.concat([raw['Home'], raw['Away']]).unique()
                    equipos = {}

                    for equipo in todos:
                        partidos = raw[(raw['Home'] == equipo) | (raw['Away'] == equipo)].tail(12)
                        GF = 0; GA = 0; MP = 0; PTS = 0
                        for _, row in partidos.iterrows():
                            if row['Home'] == equipo:
                                gf = row['HG']; ga = row['AG']
                            else:
                                gf = row['AG']; ga = row['HG']
                            GF += gf; GA += ga; MP += 1
                            if gf > ga: PTS += 3
                            elif gf == ga: PTS += 1
                        elo = 1500 + (GF - GA) * 8 + PTS * 2
                        equipos[equipo] = {"Squad": equipo, "GF": GF, "GA": GA, "MP": max(MP, 1), "Elo": elo}

                    df = pd.DataFrame(list(equipos.values()))
                    X_train, y_train = [], []
                    for _, row in raw.iterrows():
                        y = 1 if row['HG'] > row['AG'] else (0 if row['HG'] == row['AG'] else 2)
                        elo_h = equipos.get(row['Home'], {}).get('Elo', 1500)
                        elo_a = equipos.get(row['Away'], {}).get('Elo', 1500)
                        ventaja = (elo_h + 100) / max(elo_a, 1) 
                        X_train.append([1 / ventaja, ventaja])
                        y_train.append(y)

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    ml_model = LogisticRegression(class_weight='balanced')
                    ml_model.fit(X_train_scaled, y_train)

                    st.session_state.model_data = {
                        "df": df, "raw": raw, "scaler": scaler, 
                        "ml_model": ml_model, "equipos": sorted(df['Squad'].unique()), "liga": liga_seleccionada
                    }
                    st.success("¡Base de datos importada y modelo entrenado!")
                except Exception as e:
                    st.error(f"Error procesando datos: {e}")

        if st.session_state.model_data is not None:
            st.divider()
            st.subheader("Generar Pronóstico")
            c1, c2 = st.columns(2)
            with c1: loc = st.selectbox("Equipo Local", st.session_state.model_data["equipos"])
            with c2: vis = st.selectbox("Equipo Visitante", st.session_state.model_data["equipos"], index=1)

            if st.button("🤖 Analizar Partido", type="primary"):
                if loc == vis:
                    st.warning("Selecciona equipos diferentes.")
                else:
                    data = st.session_state.model_data
                    df, raw, scaler, ml_model = data["df"], data["raw"], data["scaler"], data["ml_model"]
                    
                    d_l = df[df['Squad'] == loc].iloc[0]
                    d_v = df[df['Squad'] == vis].iloc[0]
                    sede_l = obtener_sede_stats(raw, loc)
                    sede_v = obtener_sede_stats(raw, vis)

                    ataque_local = ((d_l['GF']/d_l['MP']) * 0.6) + ((sede_l['local_gf']/sede_l['local_mp']) * 0.4)
                    defensa_local = ((d_l['GA']/d_l['MP']) * 0.6) + ((sede_l['local_ga']/sede_l['local_mp']) * 0.4)
                    ataque_visita = ((d_v['GF']/d_v['MP']) * 0.6) + ((sede_v['visita_gf']/sede_v['visita_mp']) * 0.4)
                    defensa_visita = ((d_v['GA']/d_v['MP']) * 0.6) + ((sede_v['visita_ga']/sede_v['visita_mp']) * 0.4)

                    l_l = ((ataque_local + defensa_visita) / 2) * 1.18
                    l_v = ((ataque_visita + defensa_local) / 2)
                    ajuste = (d_l['Elo'] - d_v['Elo']) / 1800
                    l_l, l_v = max(l_l * (1 + ajuste), 0.1), max(l_v * (1 - ajuste), 0.1)

                    matriz = np.outer(poisson.pmf(range(10), l_l), poisson.pmf(range(10), l_v))
                    matriz /= matriz.sum()
                    prob_local, prob_empate, prob_visita = np.sum(np.tril(matriz, -1))*100, np.sum(np.diag(matriz))*100, np.sum(np.triu(matriz, 1))*100

                    goles = np.add.outer(range(10), range(10))
                    def calc_over(n): return np.sum(matriz[goles > n]) * 100
                    over25, under25, under35, over15, btts = calc_over(2.5), 100-calc_over(2.5), 100-calc_over(3.5), calc_over(1.5), np.sum(matriz[1:, 1:])*100
                    over05, over35, over45 = calc_over(0.5), calc_over(3.5), calc_over(4.5)
                    under05, under15, under45 = 100-over05, 100-over15, 100-over45

                    marcadores = []
                    for i in range(10):
                        for j in range(10): marcadores.append((matriz[i, j] * 100, i, j))
                    marcadores.sort(reverse=True)
                    texto_marcadores = ""
                    for i in range(5):
                        p, gl, gv = marcadores[i]
                        texto_marcadores += f"{i+1}. **{loc} {gl} - {gv} {vis}** ({p:.1f}%)\n\n"

                    ventaja_local = (d_l['Elo'] + 100) / max(d_v['Elo'], 1)
                    X_pred_scaled = scaler.transform([[1 / ventaja_local, ventaja_local]])
                    probs_ml = ml_model.predict_proba(X_pred_scaled)[0]
                    clases = list(ml_model.classes_)
                    prob_empate_ml = probs_ml[clases.index(0)] * 100 if 0 in clases else 0
                    prob_local_ml = probs_ml[clases.index(1)] * 100 if 1 in clases else 0
                    prob_visita_ml = probs_ml[clases.index(2)] * 100 if 2 in clases else 0

                    recomendacion = generar_recomendacion(prob_local_ml, prob_empate_ml, prob_visita_ml, over05, over15, over25, over35, over45, under25, under35, btts)

                    salida_completa = f"{loc} vs {vis}\nML: L:{prob_local_ml:.1f}% E:{prob_empate_ml:.1f}% V:{prob_visita_ml:.1f}%\nxG: {l_l:.2f}-{l_v:.2f}\nBTTS: {btts:.1f}%\nRec: {recomendacion}"
                    
                    try:
                        supabase.table("historial_apuestas").insert({
                            "user_id": st.session_state.usuario_id, "liga": data["liga"],
                            "equipo_local": loc, "equipo_visita": vis, "recomendacion": salida_completa
                        }).execute()
                    except Exception as e: st.warning(f"Error historial: {e}")

                    st.success("¡Análisis completado!")
                    with st.expander("📊 Resultados", expanded=True):
                        st.write(f"**IA ML:** L:{prob_local_ml:.1f}% | E:{prob_empate_ml:.1f}% | V:{prob_visita_ml:.1f}%")
                        st.write(f"**Recomendación:** {recomendacion}")

    # ----------------------------------------
    # PESTAÑA 2: EL HISTORIAL (MODIFICADA)
    # ----------------------------------------
    with tab_hist:
        st.header("📜 Historial de Análisis")
        
        # El botón de refrescar ahora primero elimina todo el historial del usuario
        if st.button("🔄 Refrescar e Iniciar Limpio"):
            try:
                # Borra los registros en Supabase que pertenecen a este usuario
                supabase.table("historial_apuestas").delete().eq("user_id", st.session_state.usuario_id).execute()
                st.rerun()
            except Exception as e:
                st.error(f"Error al limpiar historial: {e}")

        try:
            res = supabase.table("historial_apuestas").select("*").eq("user_id", st.session_state.usuario_id).order("fecha", desc=True).execute()
            datos = res.data
            
            if not datos:
                st.info("No hay pronósticos en el historial.")
            else:
                for fila in datos:
                    with st.expander(f"🗓️ {fila['fecha'][:10]} | {fila['equipo_local']} vs {fila['equipo_visita']}"):
                        st.code(fila['recomendacion'], language="markdown")
        except Exception as e:
            st.error(f"Error cargando historial: {e}")
