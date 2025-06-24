import json

def carregar_dados_ultimo_jogo(path='data/ultimo_jogo.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)