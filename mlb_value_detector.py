import numpy as np

class MLBValueDetector:
    def __init__(self, bankroll: float = 1000.0, kelly_multiplier: float = 0.25):
        """
        :param bankroll: Tu capital total disponible para apostar.
        :param kelly_multiplier: Factor de seguridad (fraccional). Usar 0.25 (Quarter Kelly) 
                                 es el estándar profesional para mitigar rachas negativas y varianza.
        """
        self.bankroll = bankroll
        self.kelly_multiplier = kelly_multiplier

    def decimal_to_probability(self, decimal_odds: float) -> float:
        """Convierte una cuota decimal de casa de apuestas en su probabilidad implícita."""
        if decimal_odds <= 1.0:
            return 1.0
        return 1.0 / decimal_odds

    def calculate_expected_value(self, model_prob: float, decimal_odds: float) -> float:
        """
        Calcula el Valor Esperado (EV) porcentual de una apuesta.
        Fórmula EV = (Probabilidad del Modelo * Cuota Decimal) - 1
        """
        ev = (model_prob * decimal_odds) - 1.0
        return round(ev * 100, 2) # Expresado en porcentaje

    def calculate_kelly_stake(self, model_prob: float, decimal_odds: float) -> float:
        """
        Calcula el porcentaje óptimo del bankroll a apostar usando el Criterio de Kelly Fraccional.
        Fórmula de Kelly: f* = (p * b - q) / b
        Donde:
        - p = probabilidad de ganar (modelo)
        - q = probabilidad de perder (1 - p)
        - b = cuota neta (decimal_odds - 1)
        """
        if decimal_odds <= 1.0:
            return 0.0
            
        b = decimal_odds - 1.0
        q = 1.0 - model_prob
        
        kelly_fraction = (model_prob * b - q) / b
        
        # Si el valor es negativo, no hay apuesta (EV negativo)
        if kelly_fraction <= 0:
            return 0.0
            
        # Aplicamos el multiplicador de seguridad fraccional
        final_stake_percentage = kelly_fraction * self.kelly_multiplier
        return round(final_stake_percentage * 100, 2) # Porcentaje del bankroll

    def analyze_market_matchup(self, team_name: str, model_win_prob: float, bookmaker_decimal_odds: float) -> dict:
        """
        Analiza si un mercado específico ofrece una apuesta de valor (*Value Bet*).
        """
        implied_prob = self.decimal_to_probability(bookmaker_decimal_odds)
        ev_percentage = self.calculate_expected_value(model_win_prob, bookmaker_decimal_odds)
        stake_pct = self.calculate_kelly_stake(model_win_prob, bookmaker_decimal_odds)
        
        is_value_bet = ev_percentage > 0
        recommended_amount = (self.bankroll * stake_pct) / 100.0

        return {
            "team": team_name,
            "model_probability": round(model_win_prob * 100, 2),
            "bookmaker_odds": bookmaker_decimal_odds,
            "implied_market_probability": round(implied_prob * 100, 2),
            "expected_value_ev": f"{ev_percentage}%",
            "is_value_bet": is_value_bet,
            "recommended_stake_pct": f"{stake_pct}%",
            "recommended_amount_usd": round(recommended_amount, 2)
        }

# Prueba rápida del módulo de valor
if __name__ == "__main__":
    detector = MLBValueDetector(bankroll=2000.0, kelly_multiplier=0.25)
    
    # Ejemplo: Nuestro modelo dice que los Yankees tienen 58% de probabilidad de ganar.
    # La casa de apuestas ofrece una cuota decimal de 1.85 (implícitamente 54%).
    analisis = detector.analyze_market_matchup(
        team_name="New York Yankees",
        model_win_prob=0.58,
        bookmaker_decimal_odds=1.85
    )
    
    print("Análisis de Apuesta de Valor:")
    for k, v in analisis.items():
        print(f"  - {k}: {v}")