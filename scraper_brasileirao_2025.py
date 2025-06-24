import tls_client
import pandas as pd
import time
import os

# Variável de progresso global
progresso_atual = 0

# Sessão segura com TLS
session = tls_client.Session(client_identifier="chrome_120")

def corrigir_caracteres(texto):
    if isinstance(texto, str):
        try:
            return texto.encode("latin1").decode("utf-8")
        except:
            return texto
    return texto

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

def get_teams_serie_a_2025():
    url = "https://api.sofascore.com/api/v1/unique-tournament/325/season/72034/teams"
    r = session.get(url)
    if r.status_code == 200:
        teams = r.json().get("teams", [])
        return [(corrigir_caracteres(t["name"]), t["id"]) for t in teams]
    return []

def get_jogos_do_time(team_name, team_id):
    url = f"https://api.sofascore.com/api/v1/team/{team_id}/events/last/0"
    r = session.get(url)
    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()
    events = data.get("events", [])
    df = pd.json_normalize(events)

    if df.empty:
        return pd.DataFrame()

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

    for col in ["Campeonato", "Time Casa", "Time Visitante"]:
        df_final[col] = df_final[col].apply(corrigir_caracteres)

    extra_data = []
    for _, row in df_final.iterrows():
        eid = row["event_id"]
        linha = row.to_dict()
        linha["Time Referência"] = team_name

        incidents = get_event_incidents(eid)
        linha["Num Gols"] = len([i for i in incidents if i.get("incidentType") == "goal"])
        linha["Cartões Amarelos"] = len([i for i in incidents if i.get("incidentType") == "yellowCard"])
        linha["Cartões Vermelhos"] = len([i for i in incidents if i.get("incidentType") == "redCard"])

        stats = get_event_statistics(eid)
        if stats:
            for group in stats[0].get("groups", []):
                nome = corrigir_caracteres(group.get("name", ""))
                if nome and "home" in group and "away" in group:
                    linha[f"{nome} - {row['Time Casa']}"] = group["home"]
                    linha[f"{nome} - {row['Time Visitante']}"] = group["away"]

        extra_data.append(linha)
        time.sleep(1)

    return pd.DataFrame(extra_data)

def rodar_scraper_com_progresso():
    global progresso_atual
    progresso_atual = 0
    todos_dados = []

    times = get_teams_serie_a_2025()
    total = len(times)

    if total == 0:
        progresso_atual = 100
        return

    for idx, (name, tid) in enumerate(times):
        df_time = get_jogos_do_time(name, tid)
        if not df_time.empty:
            todos_dados.append(df_time)
        progresso_atual = int(((idx + 1) / total) * 100)

    df_final = pd.concat(todos_dados, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    df_final.to_csv("data/brasileirao_serieA_2025_completo.csv", index=False, encoding="utf-8-sig")
    progresso_atual = 100
