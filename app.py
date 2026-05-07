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
    # Recuerda que esto lee de tu archivo .streamlit/secrets.toml
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
    # --- MENÚ LATERAL (SIDEBAR) ---
    st.sidebar.title("🎲 El Gordo Picks")
    st.sidebar.success("Sesión Activa")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.usuario_id = None
        st.session_state.model_data = None
        st.rerun()

    # --- PESTAÑAS PRINCIPALES ---
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

        # Cambio de nombre del botón a Importar
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

                    equipos = {}
                    todos = pd.concat([raw['Home'], raw['Away']]).unique()

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
                    equipos_lista = sorted(df['Squad'].unique())

                    X_train = []
                    y_train = []
                    for _, row in raw.iterrows():
                        if row['HG'] > row['AG']: y = 1
                        elif row['HG'] == row['AG']: y = 0
                        else: y = 2
                            
                        elo_h = equipos.get(row['Home'], {}).get('Elo', 1500)
                        elo_a = equipos.get(row['Away'], {}).get('Elo', 1500)
                        ventaja = (elo_h + 100) / max(elo_a, 1) 
                        X_train.append([1 / ventaja, ventaja])
                        y_train.append(y)

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    ml_model = LogisticRegression(class_weight='balanced')
                    ml_model.fit(X_train_scaled, y_train)

                    # Guardar modelo en sesión
                    st.session_state.model_data = {
                        "df": df, "raw": raw, "scaler": scaler, 
                        "ml_model": ml_model, "equipos": equipos_lista, "liga": liga_seleccionada
                    }
                    st.success("¡Base de datos importada y modelo entrenado con éxito!")
                except Exception as e:
                    st.error(f"Error procesando datos: {e}")

        # Si el modelo ya está entrenado, mostramos los selectores de equipos
        if st.session_state.model_data is not None:
            st.divider()
            st.subheader("Generar Pronóstico")
            
            c1, c2 = st.columns(2)
            with c1:
                loc = st.selectbox("Equipo Local", st.session_state.model_data["equipos"])
            with c2:
                vis = st.selectbox("Equipo Visitante", st.session_state.model_data["equipos"], index=1)

            if st.button("🤖 Analizar Partido", type="primary"):
                if loc == vis:
                    st.warning("Selecciona equipos diferentes.")
                else:
                    data = st.session_state.model_data
                    df = data["df"]; raw = data["raw"]; scaler = data["scaler"]; ml_model = data["ml_model"]
                    
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
                    l_l = max(l_l * (1 + ajuste), 0.1)
                    l_v = max(l_v * (1 - ajuste), 0.1)

                    matriz = np.outer(poisson.pmf(range(10), l_l), poisson.pmf(range(10), l_v))
                    matriz = matriz / matriz.sum()

                    prob_local = np.sum(np.tril(matriz, -1)) * 100
                    prob_empate = np.sum(np.diag(matriz)) * 100
                    prob_visita = np.sum(np.triu(matriz, 1)) * 100

                    goles = np.add.outer(range(10), range(10))
                    def calc_over(n): return np.sum(matriz[goles > n]) * 100

                    over05 = calc_over(0.5); under05 = 100 - over05
                    over15 = calc_over(1.5); under15 = 100 - over15
                    over25 = calc_over(2.5); under25 = 100 - over25
                    over35 = calc_over(3.5); under35 = 100 - over35
                    over45 = calc_over(4.5); under45 = 100 - over45
                    btts = np.sum(matriz[1:, 1:]) * 100

                    marcadores = []
                    for i in range(10):
                        for j in range(10):
                            marcadores.append((matriz[i, j] * 100, i, j))
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

                    recomendacion = generar_recomendacion(
                        prob_local_ml, prob_empate_ml, prob_visita_ml,
                        over05, over15, over25, over35, over45, under25, under35, btts
                    )

                    # Texto limpio para la base de datos (se mantiene igual para el registro)
                    salida_completa = f"""==============================
{loc} vs {vis}
==============================
--- PROBABILIDADES 1X2 (SKLEARN ML) ---
Local: {prob_local_ml:.1f}% | Empate: {prob_empate_ml:.1f}% | Visita: {prob_visita_ml:.1f}%

--- PROBABILIDADES 1X2 (POISSON) ---
Local: {prob_local:.1f}% | Empate: {prob_empate:.1f}% | Visita: {prob_visita:.1f}%

--- GOLES ESPERADOS (xG) ---
{loc}: {l_l:.2f}
{vis}: {l_v:.2f}

--- OVER / UNDER ---
+0.5: {over05:.1f}% | -0.5: {under05:.1f}%
+1.5: {over15:.1f}% | -1.5: {under15:.1f}%
+2.5: {over25:.1f}% | -2.5: {under25:.1f}%
+3.5: {over35:.1f}% | -3.5: {under35:.1f}%
+4.5: {over45:.1f}% | -4.5: {under45:.1f}%

Ambos Equipos Anotan (BTTS): {btts:.1f}%

--- TOP 5 MARCADORES EXACTOS ---
{texto_marcadores}
--- RECOMENDACIONES ---
{recomendacion}
"""
                    # GUARDAR EN SUPABASE AUTOMÁTICAMENTE
                    try:
                        supabase.table("historial_apuestas").insert({
                            "user_id": st.session_state.usuario_id,
                            "liga": data["liga"],
                            "equipo_local": loc,
                            "equipo_visita": vis,
                            "recomendacion": salida_completa
                        }).execute()
                    except Exception as e:
                        st.warning(f"Error guardando en historial: {e}")

                    # --- NUEVO DISEÑO VISUAL PARA PANTALLA ---
                    st.success("¡Análisis completado y guardado en tu historial!")
                    st.markdown(f"### 🏟️ {loc} vs {vis}")
                    
                    with st.expander("📊 Probabilidades 1X2", expanded=True):
                        col_ia, col_po = st.columns(2)
                        with col_ia:
                            st.markdown("**🤖 Sklearn ML (IA)**")
                            st.write(f"🏠 Local: {prob_local_ml:.1f}%")
                            st.write(f"🤝 Empate: {prob_empate_ml:.1f}%")
                            st.write(f"✈️ Visita: {prob_visita_ml:.1f}%")
                        with col_po:
                            st.markdown("**📉 Poisson Tradicional**")
                            st.write(f"🏠 Local: {prob_local:.1f}%")
                            st.write(f"🤝 Empate: {prob_empate:.1f}%")
                            st.write(f"✈️ Visita: {prob_visita:.1f}%")

                    with st.expander("⚽ Goles Esperados (xG) y BTTS"):
                        st.markdown(f"**{loc}:** {l_l:.2f} xG")
                        st.markdown(f"**{vis}:** {l_v:.2f} xG")
                        st.markdown("---")
                        st.markdown(f"🔥 **Ambos Equipos Anotan (BTTS):** {btts:.1f}%")

                    with st.expander("📈 Over / Under Total de Goles"):
                        st.markdown(f"**0.5:** 🟢 + {over05:.1f}% | 🔴 - {under05:.1f}%")
                        st.markdown(f"**1.5:** 🟢 + {over15:.1f}% | 🔴 - {under15:.1f}%")
                        st.markdown(f"**2.5:** 🟢 + {over25:.1f}% | 🔴 - {under25:.1f}%")
                        st.markdown(f"**3.5:** 🟢 + {over35:.1f}% | 🔴 - {under35:.1f}%")
                        st.markdown(f"**4.5:** 🟢 + {over45:.1f}% | 🔴 - {under45:.1f}%")

                    with st.expander("🎯 Top 5 Marcadores Exactos"):
                        st.markdown(texto_marcadores)

                    with st.expander("💡 Recomendaciones de 'El Gordo'", expanded=True):
                        st.markdown(recomendacion)

    # ----------------------------------------
    # PESTAÑA 2: EL HISTORIAL
    # ----------------------------------------
    with tab_hist:
        st.header("📜 Historial de Análisis")
        if st.button("🔄 Refrescar Historial"):
            st.rerun()

        try:
            res = supabase.table("historial_apuestas").select("*").eq("user_id", st.session_state.usuario_id).order("fecha", desc=True).execute()
            datos = res.data
            
            if not datos:
                st.info("Aún no tienes pronósticos guardados en tu cuenta.")
            else:
                for fila in datos:
                    fecha_corta = fila['fecha'][:10]
                    titulo = f"🗓️ {fecha_corta} | 🏆 {fila['liga']} | ⚽ {fila['equipo_local']} vs {fila['equipo_visita']}"
                    
                    with st.expander(titulo):
                        st.code(fila['recomendacion'], language="markdown")
                        
        except Exception as e:
            st.error(f"Error cargando historial: {e}")
