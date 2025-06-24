# scraper_brasileirao_2025.py
import tls_client
import pandas as pd
import time
import os

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

def get_teams_serie_a_2025():
    print("🔍 Buscando times da Série A de 2025...")
    url = "https://api.sofascore.com/api/v1/unique-tournament/325/season/72034/teams"
    r = session.get(url)
    if r.status_code == 200:
        data = r.json()
        teams = data.get("teams", [])
        return [(team["name"], team["id"]) for team in teams]
    else:
        print("❌ Erro ao obter os times:", r.status_code)
        return []

def get_jogos_do_time(team_name, team_id):
    print(f"⚽ Coletando jogos do {team_name}...")
    url = f"https://api.sofascore.com/api/v1/team/{team_id}/events/last/0"
    r = session.get(url)
    if r.status_code != 200:
        print(f"❌ Erro ao buscar jogos do {team_name}")
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
                nome = group.get("name", None)
                if nome and "home" in group and "away" in group:
                    linha[f"{nome} - {row['Time Casa']}"] = group["home"]
                    linha[f"{nome} - {row['Time Visitante']}"] = group["away"]
        extra_data.append(linha)
        time.sleep(1)

    return pd.DataFrame(extra_data)

def main():
    todos_dados = []
    times = get_teams_serie_a_2025()
    if not times:
        print("⚠️ Nenhum time encontrado!")
        return

    for name, tid in times:
        df_time = get_jogos_do_time(name, tid)
        if not df_time.empty:
            todos_dados.append(df_time)

    df_final = pd.concat(todos_dados, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    df_final.to_csv("data/brasileirao_serieA_2025_completo.csv", index=False)
    print("✅ Arquivo salvo com sucesso: data/brasileirao_serieA_2025_completo.csv")

if __name__ == "__main__":
    main()