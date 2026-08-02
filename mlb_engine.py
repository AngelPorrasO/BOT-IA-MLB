import numpy as np
import pandas as pd

class MLBPredictionEngine:
    """
    Motor sabermértico y de procesamiento unificado para AnRoMe MLB Predictor Pro.
    Maneja cálculos avanzados, normalización, props de ponches/hits y Criterio de Kelly.
    """
    
    def __init__(self):
        self.version = "3.6-Institutional-Props"
        self.league_avg_xfip = 4.15
        self.league_avg_woba = 0.315

    def normalize_metrics(self, xwoba, xfip, barrel, k_rate):
        clean_xwoba = max(0.200, min(0.500, float(xwoba)))
        clean_xfip = max(2.00, min(7.00, float(xfip)))
        clean_barrel = max(0.010, min(0.300, float(barrel)))
        clean_k_rate = max(0.050, min(0.500, float(k_rate)))
        
        return {
            "xwoba": clean_xwoba,
            "xfip": clean_xfip,
            "barrel": clean_barrel,
            "k_rate": clean_k_rate
        }

    def calculate_expected_runs_multiplier(self, team_woba, pitcher_xfip, park_factor, environmental_factor):
        pitcher_ratio = self.league_avg_xfip / max(2.5, pitcher_xfip)
        base_factor = (team_woba / self.league_avg_woba) * pitcher_ratio
        final_multiplier = base_factor * park_factor * environmental_factor
        return max(0.5, min(2.0, final_multiplier))

    def calculate_pitcher_props(self, k_rate, xfip, opponent_xwoba, umpire_k_mod, projected_innings=5.5):
        """
        Calcula las proyecciones cuantitativas para Strikeouts (Ponches) y Hits permitidos.
        """
        base_strikeouts = (k_rate * 27) * (self.league_avg_xfip / max(3.0, xfip))
        projected_ks = round(base_strikeouts * (projected_innings / 9.0) * umpire_k_mod, 1)
        
        base_hits_per_inning = 1.0 + ((opponent_xwoba - 0.315) * 2.5) + ((xfip - 4.15) * 0.08)
        projected_hits = round(max(2.0, base_hits_per_inning * projected_innings), 1)
        
        return {
            "strikeouts": projected_ks,
            "hits_allowed": projected_hits
        }

    def evaluate_kelly_criterion(self, probability, decimal_odds, bankroll, fraction=0.25):
        if decimal_odds <= 1.0 or probability <= 0.0:
            return 0.0, 0.0
            
        b = decimal_odds - 1.0
        p = probability
        q = 1.0 - p
        
        kelly_full = (b * p - q) / b
        if kelly_full <= 0:
            return 0.0, 0.0
            
        recommended_fraction = kelly_full * fraction
        recommended_stake = recommended_fraction * bankroll
        
        return round(recommended_stake, 2), round(recommended_fraction * 100, 2)