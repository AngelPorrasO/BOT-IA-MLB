import os
import urllib.request
import json
import math
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mlb_engine import MLBPredictionEngine

# Credenciales y APIs (Se pueden leer de forma segura desde las Variables de Entorno de GitHub)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "a14da6e6032fa3ad0587bf35475987ff")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "tu_api_key_openweathermap_aqui")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "TU_TOKEN_DE_TELEGRAM")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "TU_CHAT_ID")
BANKROLL = float(os.getenv("BANKROLL", 1000.0))
KELLY_FRACTION = 0.25

engine = MLBPredictionEngine()

def fetch_mlb_schedule(date_str):
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&date={date_str}&hydrate=probablePitcher,team,venue"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            games_list = []
            if 'dates' in data and len(data['dates']) > 0:
                for game in data['dates'][0].get('games', []):
                    home_team = game['teams']['home']['team']['name']
                    away_team = game['teams']['away']['team']['name']
                    venue_name = game.get('venue', {}).get('name', 'Estadio Neutral')
                    hp = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'Abridor Local')
                    ap = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'Abridor Visitante')
                    
                    games_list.append({
                        "matchup": f"{away_team} @ {home_team}",
                        "home_team": home_team,
                        "away_team": away_team,
                        "venue": venue_name,
                        "home_pitcher": hp,
                        "away_pitcher": ap
                    })
            return games_list
    except Exception as e:
        print(f"Error al obtener partidos de la MLB: {e}")
        return []

def fetch_live_market_odds(api_key):
    if not api_key:
        return {}
    try:
        url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={api_key}&regions=us&markets=h2h,totals&oddsFormat=decimal"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            odds_dict = {}
            for event in data:
                home_team = event.get('home_team')
                bookmakers = event.get('bookmakers', [])
                if bookmakers:
                    markets = bookmakers[0].get('markets', [])
                    h2h_odds = {}
                    for m in markets:
                        if m['key'] == 'h2h':
                            for outcome in m['outcomes']:
                                h2h_odds[outcome['name']] = outcome['price']
                    odds_dict[home_team] = {"h2h": h2h_odds}
            return odds_dict
    except Exception:
        return {}

def fetch_real_time_weather(city_name, api_key):
    if not api_key or api_key == "tu_api_key_openweathermap_aqui":
        return 1.0, "Clima Neutro"
    try:
        formatted_city = city_name.replace(" ", "%20")
        url = f"https://api.openweathermap.org/data/2.5/weather?q={formatted_city}&units=metric&appid={api_key}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            w_data = json.loads(response.read().decode())
            wind_speed_ms = w_data.get("wind", {}).get("speed", 3.0)
            temp_c = w_data.get("main", {}).get("temp", 22.0)
            
            wind_factor = 1.0
            if wind_speed_ms > 6.0: wind_factor = 1.06
            elif wind_speed_ms < 2.0: wind_factor = 0.97
            if temp_c > 30: wind_factor += 0.03
            elif temp_c < 10: wind_factor -= 0.03
                
            return round(wind_factor, 3), f"Temp: {temp_c}°C | Viento: {wind_speed_ms}m/s"
    except Exception:
        return 1.0, "Clima Neutro (Error API)"

@st.cache_resource if 'st' in globals() else lambda f: f
def get_trained_hybrid_ml_model():
    np.random.seed(42)
    X_train = np.random.rand(1000, 6)
    y_train = (X_train[:, 0] * 0.3 + X_train[:, 1] * 0.3 - X_train[:, 2] * 0.2 + X_train[:, 4] * 0.2 + np.random.normal(0, 0.1, 1000) > 0.5).astype(int)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def predict_with_random_forest(diff_wrc, diff_woba, diff_era, diff_fatigue, diff_l10, weather_factor):
    model = get_trained_hybrid_ml_model()
    features = np.array([[diff_wrc, diff_woba, diff_era, diff_fatigue, diff_l10, weather_factor]])
    probs = model.predict_proba(features)[0]
    return probs[1]

def send_telegram_message(token, chat_id, message):
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode())
            return res_data.get("ok", False)
    except Exception:
        return False

# Base de respaldo de estadísticas de equipos
FALLBACK_TEAMS = {
    "Los Angeles Dodgers": {"wrc_plus": 118, "woba": 0.338, "bullpen_era": 3.45, "bp_vs_l": 3.20, "bp_vs_r": 3.60, "fatigue": 1.0, "l10": 0.70, "pitcher": "Tyler Glasnow", "era": 3.15, "k9": 11.2, "hand": "R", "city": "Los Angeles"},
    "New York Yankees": {"wrc_plus": 115, "woba": 0.335, "bullpen_era": 3.55, "bp_vs_l": 3.35, "bp_vs_r": 3.70, "fatigue": 1.05, "l10": 0.65, "pitcher": "Gerrit Cole", "era": 2.95, "k9": 10.8, "hand": "R", "city": "New York"},
    "Philadelphia Phillies": {"wrc_plus": 114, "woba": 0.332, "bullpen_era": 3.40, "bp_vs_l": 3.15, "bp_vs_r": 3.55, "fatigue": 1.0, "l10": 0.65, "pitcher": "Zack Wheeler", "era": 2.75, "k9": 10.2, "hand": "R", "city": "Philadelphia"},
    "Baltimore Orioles": {"wrc_plus": 113, "woba": 0.333, "bullpen_era": 3.65, "bp_vs_l": 3.50, "bp_vs_r": 3.75, "fatigue": 1.02, "l10": 0.62, "pitcher": "Corbin Burnes", "era": 2.90, "k9": 10.4, "hand": "R", "city": "Baltimore"}
}

def get_team_stats(team_name):
    for key in FALLBACK_TEAMS:
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return FALLBACK_TEAMS[key]
    return {"wrc_plus": 100, "woba": 0.312, "bullpen_era": 4.00, "bp_vs_l": 3.8, "bp_vs_r": 4.1, "fatigue": 1.0, "l10": 0.50, "pitcher": "Abridor Estándar", "era": 3.80, "k9": 8.5, "hand": "R", "city": "New York"}

def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Ejecutando análisis autónomo para la fecha: {today_str}")
    
    games = fetch_mlb_schedule(today_str)
    live_odds = fetch_live_market_odds(ODDS_API_KEY)
    
    if not games:
        print("No se encontraron partidos para hoy.")
        return

    report_lines = [f"🚨 *ANROME PREDICTOR PRO - REPORTE AUTOMÁTICO* 🚨\n📅 *Fecha:* `{today_str}`\n"]

    for game in games:
        h_name = game["home_team"]
        a_name = game["away_team"]
        
        h_stats = get_team_stats(h_name)
        a_stats = get_team_stats(a_name)
        
        weather_factor, weather_desc = fetch_real_time_weather(h_stats.get("city", "New York"), WEATHER_API_KEY)
        
        effective_home_bp = h_stats["bp_vs_l"] if a_stats["hand"] == 'L' else h_stats["bp_vs_r"]
        effective_away_bp = a_stats["bp_vs_l"] if h_stats["hand"] == 'L' else a_stats["bp_vs_r"]

        home_data = {"wrc_plus": h_stats["wrc_plus"], "woba": h_stats["woba"], "bullpen_era": effective_home_bp, "bullpen_workload_fatigue": h_stats["fatigue"], "last_10_win_pct": h_stats["l10"]}
        away_data = {"wrc_plus": a_stats["wrc_plus"], "woba": a_stats["woba"], "bullpen_era": effective_away_bp, "bullpen_workload_fatigue": a_stats["fatigue"], "last_10_win_pct": a_stats["l10"]}
        
        result = engine.simulate_advanced_matchup(
            home_stats=home_data, away_stats=away_data,
            home_pitcher={"era": h_stats["era"], "k_per_9": h_stats["k9"]},
            away_pitcher={"era": a_stats["era"], "k_per_9": a_stats["k9"]},
            park_factor=1.0, home_pitcher_hand=h_stats["hand"], away_pitcher_hand=a_stats["hand"],
            umpire_data={"runs_modifier": 1.0, "k_modifier": 1.0},
            weather_data={"wind_factor": weather_factor}, iterations=20000
        )

        diff_wrc = (h_stats["wrc_plus"] - a_stats["wrc_plus"]) / 50.0
        diff_woba = (h_stats["woba"] - a_stats["woba"]) / 0.1
        diff_era = (h_stats["era"] - a_stats["era"]) / 2.0
        diff_fatigue = (h_stats["fatigue"] - a_stats["fatigue"])
        diff_l10 = (h_stats["l10"] - a_stats["l10"])
        
        rf_home_prob = predict_with_random_forest(diff_wrc, diff_woba, diff_era, diff_fatigue, diff_l10, weather_factor)
        hybrid_home_prob = (result['home_win_probability'] * 0.70) + (rf_home_prob * 0.30)
        
        # Validación de Cuotas y +EV en el mercado
        market_game = live_odds.get(h_name, {}).get("h2h", {})
        market_home_odds = market_game.get(h_name, None)
        
        ev_text = "Sin valor +EV detectado"
        if market_home_odds:
            implied = 1 / market_home_odds
            edge = (hybrid_home_prob - implied) * 100
            if edge > 0:
                ev_text = f"🔥 *¡+EV Local!* Edge: `+{edge:.1f}%` (Cuota: `{market_home_odds}`)"

        game_summary = (
            f"⚾ *{a_name} @ {h_name}*\n"
            f"• Prob. Híbrida Local: `{hybrid_home_prob*100:.1f}%`\n"
            f"• Totales Proyectados: `{result['dynamic_total_line']}` (Over: `{result['over_probability']*100:.1f}%`)\n"
            f"• {ev_text}\n"
            f"• Clima: {weather_desc}\n\n"
        )
        report_lines.append(game_summary)

    final_message = "\n".join(report_lines)
    success = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, final_message)
    if success:
        print("Reporte automático enviado con éxito a Telegram.")
    else:
        print("Error al enviar el reporte a Telegram.")

if __name__ == "__main__":
    main()