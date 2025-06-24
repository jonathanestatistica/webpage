import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import callback_context
import os

def load_data():
    csv_path = 'data/jogos_atletico.csv'
    if not os.path.exists(csv_path):
        from dash_utils.scraper import baixar_dados_galo
        baixar_dados_galo()
    df = pd.read_csv(csv_path)
    df['Data e Hora'] = pd.to_datetime(df['Data e Hora'])
    return df

def register_dash_app(flask_app):
    dash_app = Dash(__name__, server=flask_app, url_base_pathname='/dash/desempenho_galo/', external_stylesheets=[dbc.themes.BOOTSTRAP])
    df = load_data()

    dash_app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Desempenho do Galo", className="text-center text-primary"))
        ]),
        html.Hr(),

        dbc.Row([
            dbc.Col([
                html.Label("Tipo de Jogo:"),
                dcc.Dropdown(
                    id='tipo-jogo-dropdown',
                    options=[
                        {'label': 'Todos', 'value': 'todos'},
                        {'label': 'Casa', 'value': 'casa'},
                        {'label': 'Fora', 'value': 'fora'}
                    ],
                    value='todos',
                    clearable=False
                ),
            ], width=3),

            dbc.Col([
                html.Label("Filtrar por Campeonato:"),
                dcc.Dropdown(
                    id='campeonato-dropdown',
                    options=[{'label': c, 'value': c} for c in sorted(df['Campeonato'].unique())],
                    value=None,
                    placeholder='Todos os campeonatos'
                )
            ], width=4),

            dbc.Col([
                html.Label("Filtrar por Partida:"),
                dcc.Dropdown(
                    id='partida-dropdown',
                    options=[{'label': f"{row['Time Casa']} x {row['Time Visitante']} - {row['Data e Hora'].strftime('%d/%m/%Y')}", 'value': row['event_id']} for _, row in df.iterrows()],
                    value=None,
                    placeholder='Todas as partidas'
                )
            ], width=5)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Button("Atualizar Dados do Galo", id='atualizar-btn', color='primary', className='mb-3'),
                html.Div(id='atualizar-msg', className='text-success')
            ])
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='grafico-gols'))
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='grafico-tabela'))
        ])
    ], fluid=True)

    @dash_app.callback(
        Output('grafico-gols', 'figure'),
        Output('grafico-tabela', 'figure'),
        Input('tipo-jogo-dropdown', 'value'),
        Input('campeonato-dropdown', 'value'),
        Input('partida-dropdown', 'value')
    )
    def atualizar_graficos(tipo, campeonato, partida):
        dff = load_data()

        if tipo == 'casa':
            dff = dff[dff['Time Casa'] == 'Atlético Mineiro']
        elif tipo == 'fora':
            dff = dff[dff['Time Visitante'] == 'Atlético Mineiro']

        if campeonato:
            dff = dff[dff['Campeonato'] == campeonato]

        if partida:
            dff = dff[dff['event_id'] == partida]

        fig_gols = px.line(dff.sort_values("Data e Hora"), x="Data e Hora", y="Num Gols", title="Evolução dos Gols por Jogo")
        fig_tabela = px.bar(dff, x="Data e Hora", y=["Gols Casa", "Gols Visitante"], barmode="group", title="Comparativo de Gols em Cada Partida")

        return fig_gols, fig_tabela

    @dash_app.callback(
        Output('atualizar-msg', 'children'),
        Input('atualizar-btn', 'n_clicks')
    )
    def atualizar_dados(n):
        if not n:
            raise PreventUpdate
        from dash_utils.scraper import baixar_dados_galo
        baixar_dados_galo()
        return "Dados atualizados com sucesso!"

