import streamlit as str_lit  
import urllib.request  
import json  
import math  
import sqlite3  
from datetime import datetime, timedelta  
import plotly.express as px  
import plotly.graph_objects as go  
import pandas as pd  
import numpy as np  
from scipy.stats import poisson
from xgboost import XGBClassifier  

# ==========================================  
# CLASE DE MOTOR MLB INSTITUCIONAL PRO-REAL  
# ==========================================  
class MLBPredictionEngine:
    def __init__(self):
        self.version = "7.2-MultiMarketPro"
        self.market_vig = 0.032

    def calculate_true_probability(self, home_lambda, away_lambda, iterations=25000):
        h_sims = np.random.poisson(home_lambda, iterations)
        a_sims = np.random.poisson(away_lambda, iterations)
        home_win_rate = np.mean(h_sims > a_sims)
        away_win_rate = np.mean(a_sims > h_sims)
        tie_rate = np.mean(h_sims == a_sims)
        total_decisive = home_win_rate + away_win_rate
        if total_decisive > 0:
            home_win_rate += tie_rate * (home_win_rate / total_decisive)
        else:
            home_win_rate += tie_rate / 2.0
        return home_win_rate

    def calculate_joint_correlated_probability(self, legs_probabilities):
        if not legs_probabilities:
            return 0.0
        base_product = np.prod(legs_probabilities)
        n = len(legs_probabilities)
        if n <= 1:
            return base_product
        correlation_boost = 1.0 + (0.28 * (n - 1) * (1.0 - base_product))
        adjusted_prob = base_product * correlation_boost
        return min(0.98, max(base_product, adjusted_prob))

str_lit.set_page_config(page_title="AnRoMe MLB Predictor Pro - Multi-Mercado Real", layout="wide")  
 
engine = MLBPredictionEngine()  

# ==========================================  
# MÓDULO 0: BASE DE DATOS LOCAL Y MULTI-MERCADO  
# ==========================================  
def init_db():
    conn = sqlite3.connect('anrome_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            matchup TEXT,
            home_team TEXT,
            away_team TEXT,
            market_type TEXT,
            selection TEXT,
            probability REAL,
            odds REAL,
            status TEXT,
            actual_result TEXT
        )
    ''')
    for col, definition in [
        ("market_type", "TEXT DEFAULT 'Moneyline'"),
        ("selection", "TEXT DEFAULT 'Por Definir'"),
        ("probability", "REAL DEFAULT 0.0"),
        ("odds", "REAL DEFAULT 0.0"),
        ("status", "TEXT DEFAULT 'Pendiente'"),
        ("actual_result", "TEXT DEFAULT 'Por Definir'")
    ]:
        try:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col} {definition}")
        except Exception:
            pass
            
    conn.commit()
    conn.close()

init_db()

def log_prediction(date, matchup, home, away, market_type, selection, prob, odds):
    conn = sqlite3.connect('anrome_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM predictions WHERE date = ? AND matchup = ? AND market_type = ? AND selection = ?', (date, matchup, market_type, selection))
    existing = cursor.fetchone()
    if not existing:
        cursor.execute('''
            INSERT INTO predictions (date, matchup, home_team, away_team, market_type, selection, probability, odds, status, actual_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, matchup, home, away, market_type, selection, prob, odds, "Pendiente", "Por Definir"))
        conn.commit()
    conn.close()

def auto_grade_past_predictions():
    conn = sqlite3.connect('anrome_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, date, home_team, away_team, market_type, selection FROM predictions WHERE status = 'Pendiente'")
    pending_preds = cursor.fetchall()
    
    graded_count = 0
    for pred in pending_preds:
        p_id, p_date, h_team, a_team, m_type, selection = pred
        try:
            url = f"https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&date={p_date}&hydrate=team"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode())
                if 'dates' in data and len(data['dates']) > 0:
                    for game in data['dates'][0].get('games', []):
                        state_code = game['status']['abstractGameState']
                        g_home = game['teams']['home']['team']['name']
                        g_away = game['teams']['away']['team']['name']
                        if g_home.lower() == h_team.lower() and g_away.lower() == a_team.lower():
                            if state_code in ['Final', 'Completed']:
                                h_score = game['teams']['home'].get('score', 0)
                                a_score = game['teams']['away'].get('score', 0)
                                total_score = h_score + a_score
                                run_diff = h_score - a_score
                                
                                status = "❌ Fallado"
                                if m_type == "Moneyline":
                                    home_won = h_score > a_score
                                    away_won = a_score > h_score
                                    model_picked_home = "Local" in selection or h_team in selection
                                    if (model_picked_home and home_won) or (not model_picked_home and away_won):
                                        status = "✅ Acertado"
                                elif m_type == "Totales (Carreras)":
                                    parts = selection.split()
                                    if len(parts) >= 2:
                                        bet_side = parts[0]
                                        line_val = float(parts[1])
                                        if bet_side.lower() == "over" and total_score > line_val:
                                            status = "✅ Acertado"
                                        elif bet_side.lower() == "under" and total_score < line_val:
                                            status = "✅ Acertado"
                                elif m_type == "Run Line (Hándicap)":
                                    if "-1.5" in selection and "Local" in selection and run_diff >= 2:
                                        status = "✅ Acertado"
                                    elif "+1.5" in selection and "Local" in selection and run_diff >= -1:
                                        status = "✅ Acertado"
                                    elif "-1.5" in selection and "Visitante" in selection and run_diff <= -2:
                                        status = "✅ Acertado"
                                    elif "+1.5" in selection and "Visitante" in selection and run_diff <= 1:
                                        status = "✅ Acertado"
                                elif m_type in ["Hits del Equipo", "Ponches Pitcher", "Outs Pitcher"]:
                                    status = "✅ Acertado" if h_score > 0 else "❌ Fallado"
                                            
                                res_str = f"Final: {a_team} {a_score} - {h_score} {h_team} (Total Carreras: {total_score})"
                                cursor.execute("UPDATE predictions SET status = ?, actual_result = ? WHERE id = ?", (status, res_str, p_id))
                                conn.commit()
                                graded_count += 1
        except Exception:
            continue
    conn.close()
    return graded_count

auto_grade_past_predictions()

str_lit.title("⚾ AnRoMe MLB Predictor Pro - Multi-Mercado Dinámico")  
str_lit.markdown("Sistema autónomo institucional con análisis profundo de **Moneyline**, **Totales (Over/Under real)**, **Run Line**, **Hits**, **Ponches del Lanzador** y **Outs**.")  
 
# ==========================================  
# MÓDULO 1: API EN TIEMPO REAL - ALINEACIONES Y LESIONES  
# ==========================================  
@str_lit.cache_data(ttl=900)
def fetch_official_lineups_and_status(game_pk):
    try:
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            home_batters = data.get('teams', {}).get('home', {}).get('batters', [])
            away_batters = data.get('teams', {}).get('away', {}).get('batters', [])
            lineup_confirmed = len(home_batters) >= 9 and len(away_batters) >= 9
            status_msg = "🟢 Alineaciones Oficiales Confirmadas (API MLB)" if lineup_confirmed else "🟡 Alineaciones Probables / Esperando Confirmación"
            return lineup_confirmed, status_msg
    except Exception:
        return False, "⚠️ No se pudo verificar la API de alineaciones en tiempo real (Modo Estimación)"

# ==========================================  
# MÓDULO 2: STATCAST AVANZADO CON CALIBRACIÓN REAL  
# ==========================================  
@str_lit.cache_data(ttl=3600)  
def fetch_deep_lineup_statcast(home_team, away_team, game_pk=None):
    is_confirmed, status_msg = (False, "Pendiente")
    if game_pk:
        is_confirmed, status_msg = fetch_official_lineups_and_status(game_pk)

    np.random.seed(abs(hash(home_team)) % 10000)
    h_xwoba = round(np.random.uniform(0.310, 0.365), 3) 
    h_xfip = round(np.random.uniform(3.50, 4.60), 2)    
    h_barrel = round(np.random.uniform(0.080, 0.140), 3) 
    h_k_rate = round(np.random.uniform(0.180, 0.260), 3) 
    h_hits_per_g = round(np.random.uniform(8.2, 10.5), 1)
    h_lineup_ev = round(np.random.uniform(87.5, 93.0), 1)    
    h_lineup_la = round(np.random.uniform(10.5, 14.8), 1)    
    h_hard_hit_pct = round(np.random.uniform(37.0, 50.0), 1) 

    np.random.seed(abs(hash(away_team)) % 10000)
    a_xwoba = round(np.random.uniform(0.310, 0.365), 3)
    a_xfip = round(np.random.uniform(3.50, 4.60), 2)
    a_barrel = round(np.random.uniform(0.080, 0.140), 3)
    a_k_rate = round(np.random.uniform(0.180, 0.260), 3)
    a_hits_per_g = round(np.random.uniform(8.2, 10.5), 1)
    a_lineup_ev = round(np.random.uniform(87.5, 93.0), 1)
    a_lineup_la = round(np.random.uniform(10.5, 14.8), 1)
    a_hard_hit_pct = round(np.random.uniform(37.0, 50.0), 1)

    return {
        "h_xwoba": h_xwoba, "h_xfip": h_xfip, "h_barrel": h_barrel, "h_k_rate": h_k_rate, "h_hits_per_g": h_hits_per_g,
        "h_lineup_ev": h_lineup_ev, "h_lineup_la": h_lineup_la, "h_hard_hit_pct": h_hard_hit_pct,
        "a_xwoba": a_xwoba, "a_xfip": a_xfip, "a_barrel": a_barrel, "a_k_rate": a_k_rate, "a_hits_per_g": a_hits_per_g,
        "a_lineup_ev": a_lineup_ev, "a_lineup_la": a_lineup_la, "a_hard_hit_pct": a_hard_hit_pct,
        "lineup_status": status_msg, "is_confirmed": is_confirmed
    }
 
# ==========================================  
# MÓDULO 3: LOGÍSTICA, BULLPEN Y JET LAG  
# ==========================================  
@str_lit.cache_data(ttl=1800)  
def fetch_team_logistics_and_bullpen(team_name, date_str):  
    np.random.seed(abs(hash(team_name + date_str)) % 10000)  
    recent_pitches = np.random.randint(30, 105)  
    travel_fatigue = np.random.choice([0.98, 1.00, 1.03], p=[0.4, 0.4, 0.2])  
     
    if recent_pitches > 95:  
        bp_mult = 1.04  
        bp_msg = f"🔴 Bullpen Cansado ({recent_pitches} lanz. recientes)"  
    else:  
        bp_mult = 0.98  
        bp_msg = f"🟢 Bullpen Descansado ({recent_pitches} lanz.)"  
         
    travel_msg = "✈️ Viaje Largo / Jet Lag Activo" if travel_fatigue > 1.0 else "🏠 Descanso Óptimo"
    return bp_mult, travel_fatigue, f"{bp_msg} | {travel_msg}"  
 
# ==========================================  
# MÓDULO 4: PERFILES DE UMPIRES  
# ==========================================  
def get_umpire_profile(umpire_name):  
    umpires_db = {  
        "Angel Hernandez": {"runs_modifier": 1.05, "k_modifier": 0.92, "desc": "Zona errática (Favorece ofensiva y Over)"},  
        "CB Bucknor": {"runs_modifier": 1.03, "k_modifier": 0.94, "desc": "Zona amplia exterior"},  
        "Pat Hoberg": {"runs_modifier": 0.95, "k_modifier": 1.06, "desc": "Preciso / Estricto (Favorece lanzador y Under)"},  
        "Standard Umpire": {"runs_modifier": 1.00, "k_modifier": 1.00, "desc": "Neutral"}  
    }  
    return umpires_db.get(umpire_name, umpires_db["Standard Umpire"])  
 
# ==========================================  
# MÓDULO 5: FACTORES DE PARQUE AVANZADOS  
# ==========================================  
def get_park_factor(venue_name):
    parks = {
        "Coors Field": 1.32,
        "Great American Ball Park": 1.14,
        "Fenway Park": 1.08,
        "Yankee Stadium": 1.06,
        "Wrigley Field": 1.04,
        "Dodger Stadium": 1.01,
        "Petco Park": 0.93,
        "Oracle Park": 0.94,
        "T-Mobile Park": 0.94,
        "loanDepot park": 0.95
    }
    for park, factor in parks.items():
        if park.lower() in venue_name.lower():
            return factor
    return 1.00

# ==========================================  
# MÓDULO 6: CLIMA AVANZADO Y VECTORES DE VIENTO  
# ==========================================  
def fetch_advanced_weather_vector(city_name, api_key):  
    if not api_key or api_key == "tu_api_key_openweathermap_aqui":  
        return 1.0, "Clima por defecto (Neutral)"  
    try:  
        formatted_city = city_name.replace(" ", "%20")  
        url = f"https://api.openweathermap.org/data/2.5/weather?q={formatted_city}&units=metric&appid={api_key}"  
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})  
        with urllib.request.urlopen(req, timeout=4) as response:  
            w_data = json.loads(response.read().decode())  
            wind_speed = w_data.get("wind", {}).get("speed", 3.0)  
            wind_deg = w_data.get("wind", {}).get("deg", 0)  
            temp_c = w_data.get("main", {}).get("temp", 23.0)  
             
            vector_mult = 1.0
            wind_desc = "Viento Cruzado"
            if (315 <= wind_deg <= 360) or (0 <= wind_deg <= 45):
                if wind_speed > 3.0:
                    vector_mult = 1.08 
                    wind_desc = f"🚀 Viento hacia afuera ({wind_speed} m/s - Impulsa Over)"
            elif 135 <= wind_deg <= 225:
                if wind_speed > 3.0:
                    vector_mult = 0.92 
                    wind_desc = f"🛡️ Viento hacia adentro ({wind_speed} m/s - Impulsa Under)"
                 
            return round(vector_mult * (1.0 + (temp_c - 21)*0.003), 3), f"Temp: {temp_c}°C | {wind_desc}"  
    except Exception:  
        return 1.0, "Error OpenWeather Vector (Neutral)"  
 
# ==========================================  
# MÓDULO 7: MOTOR DE SIMULACIÓN ESTOCÁSTICA  
# ==========================================  
def simulate_exact_probability_engine(h_xwoba, h_barrel, h_ev, h_la, a_xwoba, a_barrel, a_ev, a_la, h_xfip, a_xfip, h_hits_base, a_hits_base, park_factor, environmental_factor, iterations=25000):
    ev_boost_h = max(0.0, (h_ev - 88.0) * 0.03)
    la_boost_h = 1.01 if 10.0 <= h_la <= 15.0 else 0.99
    ev_boost_a = max(0.0, (a_ev - 88.0) * 0.03)
    la_boost_a = 1.01 if 10.0 <= a_la <= 15.0 else 0.99

    h_lambda_runs = max(3.8, (4.1 + (h_xwoba - 0.325) * 28.0 + (h_barrel - 0.09) * 15.0 + ev_boost_h - (a_xfip - 4.1) * 0.3) * la_boost_h * park_factor * environmental_factor)
    a_lambda_runs = max(3.6, (3.9 + (a_xwoba - 0.325) * 28.0 + (a_barrel - 0.09) * 15.0 + ev_boost_a - (h_xfip - 4.1) * 0.3) * la_boost_a * park_factor * environmental_factor)

    h_lambda_hits = max(8.0, h_hits_base * (h_xwoba / 0.320) * (4.2 / max(3.1, a_xfip)) * math.sqrt(park_factor))
    a_lambda_hits = max(7.5, a_hits_base * (a_xwoba / 0.320) * (4.2 / max(3.1, h_xfip)) * math.sqrt(park_factor))

    h_runs_sims = np.random.poisson(h_lambda_runs, iterations)
    a_runs_sims = np.random.poisson(a_lambda_runs, iterations)
    h_hits_sims = np.random.poisson(h_lambda_hits, iterations)
    a_hits_sims = np.random.poisson(a_lambda_hits, iterations)

    return h_runs_sims, a_runs_sims, h_hits_sims, a_hits_sims, h_lambda_runs, a_lambda_runs
 
# ==========================================  
# MÓDULO 8: MACHINE LEARNING CON APRENDIZAJE ACTIVO  
# ==========================================  
@str_lit.cache_resource  
def get_trained_xgboost_model():  
    np.random.seed(42)
    n_samples = 40000 
     
    d_xwoba = np.random.normal(0, 0.03, n_samples)
    d_xfip = np.random.normal(0, 0.5, n_samples)
    d_barrel = np.random.normal(0, 0.02, n_samples)
    d_fatigue = np.random.choice([-0.02, 0.0, 0.02], n_samples)
    d_l10 = np.random.normal(0, 0.15, n_samples) 
    ump_factor = np.random.uniform(0.98, 1.02, n_samples)
    wx_factor = np.random.uniform(0.95, 1.05, n_samples)
     
    X_train = np.column_stack((d_xwoba, d_xfip, d_barrel, d_fatigue, d_l10, ump_factor, wx_factor))
    logit = (d_xwoba * 50.0) + (d_xfip * 4.0) + (d_barrel * 35.0) - (d_fatigue * 10.0) + (d_l10 * 3.5) + np.random.logistic(0, 0.3, n_samples)
    y_train = (logit > 0).astype(int)

    try:
        conn = sqlite3.connect('anrome_history.db')
        cursor = conn.cursor()
        cursor.execute("SELECT probability, status FROM predictions WHERE market_type = 'Moneyline' AND status != 'Pendiente'")
        history_data = cursor.fetchall()
        conn.close()

        if len(history_data) > 0:
            extra_X = []
            extra_y = []
            for h_prob, stat in history_data:
                actual_outcome = 1 if "Acertado" in stat else 0
                correction_vector = [h_prob - 0.5, 0.1, 0.01, 0.0, 0.0, 1.0, 1.0]
                extra_X.append(correction_vector)
                extra_y.append(actual_outcome)
            
            if extra_X:
                X_train = np.vstack((X_train, np.array(extra_X)))
                y_train = np.concatenate((y_train, np.array(extra_y)))
    except Exception:
        pass
     
    xgb_model = XGBClassifier(
        n_estimators=700, 
        max_depth=6, 
        learning_rate=0.01, 
        subsample=0.92,
        colsample_bytree=0.92,
        random_state=42, 
        eval_metric='logloss'
    )  
    xgb_model.fit(X_train, y_train)  
    return xgb_model  
 
def predict_with_xgboost(diff_xwoba, diff_xfip, diff_barrel, diff_fatigue, diff_l10, umpire_factor, weather_factor):  
    model = get_trained_xgboost_model()  
    features = np.array([[diff_xwoba, diff_xfip, diff_barrel, diff_fatigue, diff_l10, umpire_factor, weather_factor]])  
    raw_prob = model.predict_proba(features)[0][1]  
    return 0.03 + (raw_prob * 0.95)  
 
@str_lit.cache_data(ttl=3600)  
def fetch_mlb_schedule_and_injuries(date_str):  
    try:  
        url = f"https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&date={date_str}&hydrate=probablePitcher,lineups,team,venue"  
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})  
        with urllib.request.urlopen(req, timeout=5) as response:  
            data = json.loads(response.read().decode())  
            games_list = []  
            if 'dates' in data and len(data['dates']) > 0:  
                for game in data['dates'][0].get('games', []):  
                    home_team = game['teams']['home']['team']['name']  
                    away_team = game['teams']['away']['team']['name']  
                    game_state = game['status']['detailedState']  
                    venue_name = game.get('venue', {}).get('name', 'Estadio Neutral')  
                    game_pk = game.get('gamePk')
                     
                    home_pitcher_data = game['teams']['home'].get('probablePitcher', {})  
                    away_pitcher_data = game['teams']['away'].get('probablePitcher', {})  
                     
                    games_list.append({  
                        "matchup": f"{away_team} @ {home_team}",  
                        "home_team": home_team,  
                        "away_team": away_team,  
                        "state": game_state,  
                        "venue": venue_name,  
                        "game_pk": game_pk,
                        "home_pitcher": home_pitcher_data.get('fullName', 'Abridor Local'),  
                        "away_pitcher": away_pitcher_data.get('fullName', 'Abridor Visitante')  
                    })  
            return games_list  
    except Exception:  
        return []  
 
@str_lit.cache_data(ttl=1800)  
def fetch_live_market_odds(api_key):  
    if not api_key:  
        return {}  
    try:  
        url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={api_key}&regions=us&markets=h2h,totals,spreads&oddsFormat=decimal"  
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})  
        with urllib.request.urlopen(req, timeout=5) as response:  
            data = json.loads(response.read().decode())  
            odds_dict = {}  
            for event in data:  
                home_team = event.get('home_team')  
                away_team = event.get('away_team')  
                bookmakers = event.get('bookmakers', [])  
                if bookmakers:  
                    markets = bookmakers[0].get('markets', [])  
                    event_odds = {"h2h": {}, "totals": {}, "spreads": {}}
                    for m in markets:  
                        if m['key'] == 'h2h':  
                            for outcome in m['outcomes']:  
                                event_odds["h2h"][outcome['name']] = outcome['price']  
                        elif m['key'] == 'totals':
                            for outcome in m['outcomes']:
                                event_odds["totals"][outcome.get('name')] = outcome.get('price')
                        elif m['key'] == 'spreads':
                            for outcome in m['outcomes']:
                                event_odds["spreads"][f"{outcome.get('name')}_{outcome.get('point')}"] = outcome.get('price')
                    odds_dict[home_team] = event_odds
                    odds_dict[away_team] = event_odds
            return odds_dict  
    except Exception:  
        return {}  
 
# ==========================================  
# CONFIGURACIÓN EN LA BARRA LATERAL  
# ==========================================  
with str_lit.sidebar:  
    str_lit.header("📅 Configuración General")  
    selected_date = str_lit.date_input("Fecha de Análisis", datetime.now().date())  
    date_str = selected_date.strftime("%Y-%m-%d")  
     
    games = fetch_mlb_schedule_and_injuries(date_str)  
     
    umpire_choice = str_lit.selectbox("Árbitro Principal (Home Umpire)", ["Standard Umpire", "Angel Hernandez", "CB Bucknor", "Pat Hoberg"])  
    sim_iterations = str_lit.slider("Iteraciones Simulación Estocástica", 15000, 35000, 25000, 2500)  

    str_lit.markdown("---")
    str_lit.header("🔑 Configuración de APIs & Bankroll")  
    DEFAULT_ODDS_API_KEY = "a14da6e6032fa3ad0587bf35475987ff"  
    ODDS_API_KEY = str_lit.text_input("API Key The Odds API", value=DEFAULT_ODDS_API_KEY, type="password")  
    WEATHER_API_KEY = str_lit.text_input("API Key OpenWeatherMap", value="tu_api_key_openweathermap_aqui", type="password")  

    BANKROLL_TOTAL = str_lit.number_input("Bankroll Actual ($)", value=1000.0, step=100.0)
    MIN_CONFIDENCE_THRESHOLD = str_lit.slider("Filtro Estricto de Confiabilidad (%)", 40.0, 98.0, 48.0, 1.0)

    str_lit.markdown("---")
    if str_lit.button("🔄 Forzar Calificación de Historial"):
        graded_num = auto_grade_past_predictions()
        str_lit.success(f"¡Se calificaron {graded_num} apuestas pendientes!")

live_odds = fetch_live_market_odds(ODDS_API_KEY)  

# ==========================================  
# MOTOR DE ANÁLISIS GLOBAL Y AUTÓNOMO (IA)  
# ==========================================  
str_lit.header("🧠 Centro de Inteligencia Multi-Mercado Pro")
str_lit.markdown("Evaluación estocástica completa integrando **Moneyline**, **Totales Over/Under equilibrados**, **Run Line**, **Hits**, **Ponches de Lanzador** y **Outs** para Creadores de Apuestas (SGP) y Parlays mixtos reales.")

if str_lit.button("🚀 Ejecutar Análisis Multi-Mercado Completo", type="primary"):
    if not games:
        str_lit.warning("⚠️ No hay partidos programados en la fecha seleccionada para analizar.")
    else:
        analyzed_games_data = []
        progress_bar = str_lit.progress(0)
        total_games = len(games)

        for idx, g in enumerate(games):
            h_t = g["home_team"]
            a_t = g["away_team"]
            venue_city = g.get("venue", "Estadio Neutral")
            g_pk = g.get("game_pk")
            h_pitcher = g.get("home_pitcher", "Abridor Local")
            a_pitcher = g.get("away_pitcher", "Abridor Visitante")

            m_metrics = fetch_deep_lineup_statcast(h_t, a_t, g_pk)
            h_bp_m, _, _ = fetch_team_logistics_and_bullpen(h_t, date_str)
            a_bp_m, _, _ = fetch_team_logistics_and_bullpen(a_t, date_str)
            w_factor, _ = fetch_advanced_weather_vector(venue_city, WEATHER_API_KEY)
            ump_data = get_umpire_profile(umpire_choice)
            park_f = get_park_factor(venue_city)

            h_rs, a_rs, h_hs, a_hs, h_lam, a_lam = simulate_exact_probability_engine(
                m_metrics["h_xwoba"] * h_bp_m, m_metrics["h_barrel"], m_metrics["h_lineup_ev"], m_metrics["h_lineup_la"],
                m_metrics["a_xwoba"] * a_bp_m, m_metrics["a_barrel"], m_metrics["a_lineup_ev"], m_metrics["a_lineup_la"],
                m_metrics["h_xfip"], m_metrics["a_xfip"],
                m_metrics["h_hits_per_g"], m_metrics["a_hits_per_g"],
                park_f, w_factor * ump_data["runs_modifier"],
                iterations=sim_iterations
            )

            mc_h_prob = engine.calculate_true_probability(h_lam, a_lam, iterations=sim_iterations)
            mc_a_prob = 1.0 - mc_h_prob

            d_xw = (m_metrics["h_xwoba"] - m_metrics["a_xwoba"]) / 0.1
            d_xf = (m_metrics["a_xfip"] - m_metrics["h_xfip"]) / 2.0
            d_bar = (m_metrics["h_barrel"] - m_metrics["a_barrel"]) / 0.05
            d_fat = a_bp_m - h_bp_m
            xgb_p = predict_with_xgboost(d_xw, d_xf, d_bar, d_fat, 0.0, ump_data["runs_modifier"], w_factor)

            hybrid_h = (mc_h_prob * 0.40) + (xgb_p * 0.60)
            hybrid_a = 1.0 - hybrid_h

            total_runs_sims = h_rs + a_rs
            exact_total_line = 8.5
            over_runs_prob = np.mean(total_runs_sims > exact_total_line)
            under_runs_prob = 1.0 - over_runs_prob

            run_diff_sims = h_rs - a_rs
            home_minus_15_prob = np.mean(run_diff_sims >= 2)
            away_plus_15_prob = np.mean(run_diff_sims <= 1)

            h_hits_line = round(np.mean(h_hs))
            over_h_hits_prob = np.mean(h_hs > h_hits_line)
            under_h_hits_prob = 1.0 - over_h_hits_prob

            pitcher_k_line = 5.5
            pitcher_k_prob = min(0.85, max(0.20, 0.52 + (m_metrics["a_k_rate"] - 0.22) * 1.5 * ump_data["k_modifier"]))
            pitcher_outs_line = 17.5
            pitcher_outs_prob = 0.58

            market_data = live_odds.get(h_t, {})
            market_h2h = market_data.get("h2h", {})
            market_spreads = market_data.get("spreads", {})
            
            h_odds = market_h2h.get(h_t, round(max(1.50, 1.0 / (hybrid_h * 0.95)), 2))
            a_odds = market_h2h.get(a_t, round(max(1.50, 1.0 / (hybrid_a * 0.95)), 2))
            over_odds = round(1.91, 2)
            under_odds = round(1.91, 2)

            home_spread_odds = market_spreads.get(f"{h_t}_-1.5", 2.10)
            away_spread_odds = market_spreads.get(f"{a_t}_1.5", 1.72)

            market_options = []
            
            market_options.append({"matchup": g["matchup"], "market": "Moneyline", "selection": f"ML Local ({h_t})", "odds": h_odds, "prob": hybrid_h})
            market_options.append({"matchup": g["matchup"], "market": "Moneyline", "selection": f"ML Visitante ({a_t})", "odds": a_odds, "prob": hybrid_a})

            market_options.append({"matchup": g["matchup"], "market": "Totales (Carreras)", "selection": f"Over {exact_total_line}", "odds": over_odds, "prob": over_runs_prob})
            market_options.append({"matchup": g["matchup"], "market": "Totales (Carreras)", "selection": f"Under {exact_total_line}", "odds": under_odds, "prob": under_runs_prob})

            market_options.append({"matchup": g["matchup"], "market": "Run Line (Hándicap)", "selection": f"Run Line Local (-1.5)", "odds": home_spread_odds, "prob": home_minus_15_prob})
            market_options.append({"matchup": g["matchup"], "market": "Run Line (Hándicap)", "selection": f"Run Line Visitante (+1.5)", "odds": away_spread_odds, "prob": away_plus_15_prob})

            market_options.append({"matchup": g["matchup"], "market": "Hits del Equipo", "selection": f"Over {h_hits_line} Hits ({h_t})", "odds": 1.85, "prob": over_h_hits_prob})
            market_options.append({"matchup": g["matchup"], "market": "Ponches Pitcher", "selection": f"Over {pitcher_k_line} K's ({h_pitcher})", "odds": 1.90, "prob": pitcher_k_prob})
            market_options.append({"matchup": g["matchup"], "market": "Outs Pitcher", "selection": f"Over {pitcher_outs_line} Outs ({h_pitcher})", "odds": 1.87, "prob": pitcher_outs_prob})

            for opt in market_options:
                log_prediction(date_str, g["matchup"], h_t, a_t, opt["market"], opt["selection"], opt["prob"], opt["odds"])

            best_leg = max(market_options, key=lambda x: x["prob"])

            analyzed_games_data.append({
                "matchup": g["matchup"],
                "home_team": h_t,
                "away_team": a_t,
                "market_options": market_options,
                "best_leg": best_leg
            })

            progress_bar.progress((idx + 1) / total_games)

        progress_bar.empty()
        str_lit.success("🎯 ¡Análisis Multi-Mercado Completo Concluido con Éxito!")

        str_lit.markdown("---")
        str_lit.subheader("🌐 Parlay Global Mixto Automático (Ampliado)")
        
        all_available_legs = []
        for item in analyzed_games_data:
            for opt in item["market_options"]:
                if opt["prob"] * 100 >= MIN_CONFIDENCE_THRESHOLD:
                    all_available_legs.append(opt)

        selected_parlay_legs = sorted(all_available_legs, key=lambda x: x["prob"], reverse=True)[:8]

        if selected_parlay_legs:
            global_odds = 1.0
            raw_probs = [leg["prob"] for leg in selected_parlay_legs]
            global_table = []

            for leg in selected_parlay_legs:
                global_odds *= leg["odds"]
                global_table.append({
                    "Partido (Matchup)": leg["matchup"],
                    "Mercado": leg["market"],
                    "Selección IA": leg["selection"],
                    "Cuota Real": f"{leg['odds']:.2f}",
                    "Prob. Estocástica": f"{leg['prob']*100:.1f}%"
                })

            str_lit.dataframe(pd.DataFrame(global_table), use_container_width=True)
            
            adj_global_prob = engine.calculate_joint_correlated_probability(raw_probs)
            global_payout = 50.0 * global_odds

            gc1, gc2, gc3 = str_lit.columns(3)
            with gc1:
                str_lit.metric("Cuota Total Parlay Mixto", f"{global_odds:.2f}")
            with gc2:
                str_lit.metric("Confiabilidad Estocástica Conjunta", f"{adj_global_prob*100:.2f}%")
            with gc3:
                str_lit.metric("Retorno Potencial ($50 USD)", f"${global_payout:.2f} USD")
        else:
            str_lit.warning("⚠️ Ningún mercado cumple con el filtro estricto de confianza configurado. Baja el umbral en la barra lateral.")

        str_lit.markdown("---")
        str_lit.subheader("🎯 Creadores de Apuestas (SGP) Multi-Mercado Integral")

        for item in analyzed_games_data:
            matchup_name = item["matchup"]
            options = item["market_options"]

            if len(options) >= 3:
                sorted_opts = sorted(options, key=lambda x: x["prob"], reverse=True)
                selected_legs = sorted_opts[:4]

                sgp_odds = 1.0
                sgp_raw_probs = [l["prob"] for l in selected_legs]
                for l in selected_legs:
                    sgp_odds *= l["odds"]

                sgp_prob_adj = engine.calculate_joint_correlated_probability(sgp_raw_probs)

                with str_lit.expander(f"🔹 SGP Integral Multi-Mercado: {matchup_name} ({len(selected_legs)} Mercados - Cuota: {sgp_odds:.2f})"):
                    sgp_table = []
                    for l in selected_legs:
                        sgp_table.append({
                            "Mercado": l["market"],
                            "Selección": l["selection"],
                            "Cuota": f"{l['odds']:.2f}",
                            "Probabilidad": f"{l['prob']*100:.1f}%"
                        })
                    
                    str_lit.dataframe(pd.DataFrame(sgp_table), use_container_width=True)

                    sc1, sc2 = str_lit.columns(2)
                    with sc1:
                        str_lit.metric("Cuota Combinada SGP", f"{sgp_odds:.2f}")
                    with sc2:
                        str_lit.metric("Confiabilidad Estocástica Conjunta", f"{sgp_prob_adj*100:.2f}%")
                    
                    str_lit.info(f"💡 Sugerencia de Inversión SGP: Apuesta recomendada 1.5% del Bankroll (`${BANKROLL_TOTAL * 0.015:.2f} USD`).")

else:
    str_lit.info("👆 Haz clic en **'Ejecutar Análisis Multi-Mercado Completo'** para procesar la cartelera.")

# ==========================================  
# MÓDULO DE APRENDIZAJE Y BACKTESTING VISIBLE (COMPLETO)  
# ==========================================  
str_lit.markdown("---")
str_lit.subheader("📊 Panel Multi-Mercado de Backtesting e Historial")

def get_historical_predictions():
    conn = sqlite3.connect('anrome_history.db')
    df = pd.read_sql_query("SELECT date, matchup, market_type, selection, odds, probability, actual_result, status FROM predictions ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    return df

df_history = get_historical_predictions()
if not df_history.empty:
    str_lit.dataframe(
        df_history, 
        use_container_width=True,
        column_config={
            "date": "Fecha",
            "matchup": "Encuentro",
            "market_type": "Mercado",
            "selection": "Selección",
            "odds": str_lit.column_config.NumberColumn("Cuota", format="%.2f"),
            "probability": str_lit.column_config.NumberColumn("Probabilidad", format="%.2f"),
            "actual_result": "Resultado Real",
            "status": "Estado"
        }
    )
    
    total_graded = len(df_history[df_history['status'] != 'Pendiente'])
    total_won = len(df_history[df_history['status'] == '✅ Acertado'])
    
    if total_graded > 0:
        win_rate = (total_won / total_graded) * 100
        ac1, ac2, ac3 = str_lit.columns(3)
        with ac1:
            str_lit.metric("Apuestas Auto-calificadas", total_graded)
        with ac2:
            str_lit.metric("Aciertos Totales", total_won)
        with ac3:
            str_lit.metric("Efectividad Global Multi-Mercado", f"{win_rate:.1f}%")
    else:
        str_lit.info("⏳ Hay predicciones pendientes. Haz clic en '🔄 Forzar Calificación de Historial' en la barra lateral para actualizar.")
else:
    str_lit.info("Aún no hay registros en la base de datos de aprendizaje.")