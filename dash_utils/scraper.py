# dash_utils/scraper.py
import tls_client
import pandas as pd
from datetime import datetime
import time
import os

def baixar_dados_galo(salvar_em='data/jogos_atletico.csv'):
    team_id = 1977
    url = f"https://api.sofascore.com/api/v1/team/{team_id}/events/last/0"
    session = tls_client.Session(client_identifier="chrome_120")

    def get_event_incidents(event_id):
        url = f"https://api.sofascore.com/api/v1/event/{event_id}/incidents"
        r = session.get(url)
        if r.status_code == 200:
            return r.json().get("incidents", [])
        return []

    def get_event_statistics(event_id):
        url = f"https://api.sofascore.com/api/v1/event/{event_id}/statistics"
        r = session.get(url)
        if r.status_code == 200:
            return r.json().get("statistics", [])
        return []

    response = session.get(url)
    if response.status_code != 200:
        print("Erro na API:", response.status_code)
        return

    data = response.json()
    events = data.get("events", [])
    df = pd.json_normalize(events)

    df_final = df[[
        "id", "tournament.name", "startTimestamp",
        "homeTeam.name", "awayTeam.name",
        "homeScore.current", "awayScore.current",
        "status.description"
    ]].copy()

    df_final["Data e Hora"] = pd.to_datetime(df_final["startTimestamp"], unit="s")
    df_final = df_final.drop(columns=["startTimestamp"])
    df_final.columns = [
        "event_id", "Campeonato", "Time Casa", "Time Visitante",
        "Gols Casa", "Gols Visitante", "Status", "Data e Hora"
    ]

    extra_data = []
    for _, row in df_final.iterrows():
        eid = row["event_id"]
        linha = row.to_dict()

        incidents = get_event_incidents(eid)
        linha["Num Gols"] = len([i for i in incidents if i.get("incidentType") == "goal"])
        linha["Cartões Amarelos"] = len([i for i in incidents if i.get("incidentType") == "yellowCard"])
        linha["Cartões Vermelhos"] = len([i for i in incidents if i.get("incidentType") == "redCard"])

        stats = get_event_statistics(eid)
        if stats:
            for group in stats[0].get("groups", []):
                nome = group.get("name")
                if nome and "home" in group and "away" in group:
                    linha[f"{nome} - Atlético"] = group["home"]
                    linha[f"{nome} - Adversário"] = group["away"]
        extra_data.append(linha)
        time.sleep(1)

    df_final = pd.DataFrame(extra_data)
    os.makedirs(os.path.dirname(salvar_em), exist_ok=True)
    df_final.to_csv(salvar_em, index=False)
    print(f"Dados atualizados: {salvar_em}")
