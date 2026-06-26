# Backend não interativo para Matplotlib
import matplotlib
matplotlib.use('Agg')

# Bibliotecas Flask
from flask import Flask, render_template, request, jsonify, send_file

from werkzeug.utils import secure_filename

# Bibliotecas padrão
import os
import io
import base64
from math import comb, exp, factorial, ceil, sqrt

# Bibliotecas de manipulação de dados e estatísticas
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, t
import statsmodels.api as sm
import openpyxl

# Plotting
import matplotlib.pyplot as plt

import threading
import scraper_brasileirao_2025 as scraper


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard/")
def dashboard():
    return render_template("dashboard.html")

# Outras rotas
@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/contact/")
def contact():
    return render_template("contact.html")

@app.route("/videos/")
def videos():
    return render_template("videos.html")

@app.route("/codigos/")
def codigos():
    return render_template("codigos.html")

@app.route("/disciplinas/")
def disciplinas():
    return render_template("disciplina.html")

# Rota para ensino
@app.route("/ensino/")
def ensino():
    return render_template("ensino.html")

#######################################################
####### TUDO RELACIONADO À BIOLOGIA #######
#######################################################

@app.route("/ensino/biologia/")
def ensino_biologia():
    return render_template("ensino/biologia.html")

# Rota para exercícios resolvidos biologia
@app.route("/ensino/biologia/exercicios_resolvidos/")
def exercicios_resolvidos_biologia():
    return render_template("ensino/biologia/exercicios_resolvidos.html")

# Rota para listas biologia
@app.route("/ensino/biologia/listas/")
def listas_biologia():
    return render_template("ensino/biologia/listas.html")

# Rota para plano de ensino biologia
@app.route("/ensino/biologia/plano_de_ensino/")
def plano_de_ensino_biologia():
    return render_template("ensino/biologia/plano_de_ensino.html")


#######################################################
####### TUDO RELACIONADO À FISIOTERAPIA #######
#######################################################

@app.route("/ensino/fisioterapia/")
def ensino_fisioterapia():
    return render_template("ensino/fisioterapia.html")

# Rota para exercícios resolvidos fisioterapia
@app.route("/ensino/fisioterapia/exercicios_resolvidos")
def exercicios_resolvidos_fisioterapia():
    return render_template("ensino/fisioterapia/exercicios_resolvidos.html")

# Rota para listas fisioterapia
@app.route("/ensino/fisioterapia/listas")
def listas_fisioterapia():
    return render_template("ensino/fisioterapia/listas.html")

# Rota para plano de ensino fisioterapia
@app.route("/ensino/fisioterapia/plano_de_ensino")
def plano_de_ensino_fisioterapia():
    return render_template("ensino/fisioterapia/plano_de_ensino.html")


#######################################################
####### TUDO RELACIONADO À TERAPIA OCUPACIONAL #######
#######################################################

@app.route("/ensino/terapia_ocupacional/")
def ensino_terapia_ocupacional():
    return render_template("ensino/terapia_ocupacional.html")

# Rota para exercícios resolvidos terapia_ocupacional
@app.route("/ensino/terapia_ocupacional/exercicios_resolvidos")
def exercicios_resolvidos_terapia_ocupacional():
    return render_template("ensino/terapia_ocupacional/exercicios_resolvidos.html")

# Rota para listas terapia_ocupacional
@app.route("/ensino/terapia_ocupacional/listas")
def listas_terapia_ocupacional():
    return render_template("ensino/terapia_ocupacional/listas.html")

# Rota para plano de ensino terapia_ocupacional
@app.route("/ensino/terapia_ocupacional/plano_de_ensino/")
def plano_de_ensino_terapia_ocupacional():
    return render_template("ensino/terapia_ocupacional/plano_de_ensino.html")


#######################################################
####### TUDO RELACIONADO À BIOMEDICINA/FARMÁCIA #######
#######################################################

@app.route("/ensino/biomedicina_farmacia/")
def ensino_biomedicina_farmacia():
    return render_template("ensino/biomedicina_farmacia.html")

# Rota para exercícios resolvidos biomedicina_farmacia
@app.route("/ensino/biomedicina_farmacia/exercicios_resolvidos")
def exercicios_resolvidos_biomedicina_farmacia():
    return render_template("ensino/biomedicina_farmacia/exercicios_resolvidos.html")

# Rota para listas biomedicina_farmacia
@app.route("/ensino/biomedicina_farmacia/listas")
def listas_biomedicina_farmacia():
    return render_template("ensino/biomedicina_farmacia/listas.html")

# Rota para plano de ensino biomedicina_farmacia
@app.route("/ensino/biomedicina_farmacia/plano_de_ensino/")
def plano_de_ensino_biomedicina_farmacia():
    return render_template("ensino/biomedicina_farmacia/plano_de_ensino.html")


# Rota para projeto
@app.route("/projeto/")
def projeto():
    return render_template("projeto.html")

# Rota para shiny_apps
@app.route("/shiny_apps/")
def shiny_apps():
    return render_template("shiny_apps.html")

# Plot Distribution
@app.route('/plot/<dist_type>')
def plot_distribution(dist_type):
    try:
        x = list(range(11))
        y = []

        if dist_type == 'binomial':
            n = int(request.args.get('n'))
            p = float(request.args.get('p'))
            x = list(range(n + 1))
            y = [comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in x]
        elif dist_type == 'hypergeometric':
            N = int(request.args.get('N'))
            n = int(request.args.get('n'))
            r = int(request.args.get('r'))
            y = [comb(r, k) * comb(N - r, n - k) / comb(N, n) if k <= min(r, n) else 0 for k in x]
        elif dist_type == 'poisson':
            lambd = float(request.args.get('lambda'))
            y = [(exp(-lambd) * lambd ** k) / factorial(k) for k in x]

        plt.bar(x, y, color='blue', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('P(X=x)')
        plt.title(f'Distribuição {dist_type.capitalize()}')
        plt.xticks(x)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar gráfico: {str(e)}'}), 500

# Calculate Binomial Distribution
@app.route('/calculate/binomial')
def calculate_binomial():
    try:
        n = int(request.args.get('n'))
        p = float(request.args.get('p'))
        k = int(request.args.get('k'))

        prob = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        cdf = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))
        plot_url = f"/plot/binomial?n={n}&p={p}"
        return jsonify({'result': f'P(X={k})={prob:.4f}, F(X≤{k})={cdf:.4f}', 'plot_url': plot_url})
    except Exception as e:
        return jsonify({'error': f'Erro ao calcular binomial: {str(e)}'}), 500

# Calculate Hypergeometric Distribution
@app.route('/calculate/hypergeometric')
def calculate_hypergeometric():
    try:
        N = int(request.args.get('N'))
        n = int(request.args.get('n'))
        r = int(request.args.get('r'))
        k = int(request.args.get('k'))

        prob = comb(r, k) * comb(N - r, n - k) / comb(N, n)
        cdf = sum(comb(r, i) * comb(N - r, n - i) / comb(N, n) for i in range(k + 1))
        plot_url = f"/plot/hypergeometric?N={N}&n={n}&r={r}"
        return jsonify({'result': f'P(X={k})={prob:.4f}, F(X≤{k})={cdf:.4f}', 'plot_url': plot_url})
    except Exception as e:
        return jsonify({'error': f'Erro ao calcular hipergeométrica: {str(e)}'}), 500

# Calculate Poisson Distribution
@app.route('/calculate/poisson')
def calculate_poisson():
    try:
        lambd = float(request.args.get('lambda'))
        k = int(request.args.get('k'))

        prob = (exp(-lambd) * lambd ** k) / factorial(k)
        cdf = sum((exp(-lambd) * lambd ** i) / factorial(i) for i in range(k + 1))
        plot_url = f"/plot/poisson?lambda={lambd}"
        return jsonify({'result': f'P(X={k})={prob:.4f}, F(X≤{k})={cdf:.4f}', 'plot_url': plot_url})
    except Exception as e:
        return jsonify({'error': f'Erro ao calcular Poisson: {str(e)}'}), 500

@app.route('/calculate/normal', methods=['POST'])
def calculate_normal():
    try:
        data = request.json
        mean = float(data['mean'])
        variance = float(data['variance'])
        std_dev = np.sqrt(variance)
        calculation_type = data['calculation_type']

        if calculation_type == 'above':
            value = float(data['value'])
            probability = 1 - norm.cdf(value, loc=mean, scale=std_dev)
        elif calculation_type == 'below':
            value = float(data['value'])
            probability = norm.cdf(value, loc=mean, scale=std_dev)
        elif calculation_type == 'between':
            value1 = float(data['value1'])
            value2 = float(data['value2'])
            probability = norm.cdf(value2, loc=mean, scale=std_dev) - norm.cdf(value1, loc=mean, scale=std_dev)
        elif calculation_type == 'outside':
            value1 = float(data['value1'])
            value2 = float(data['value2'])
            probability = 1 - (norm.cdf(value2, loc=mean, scale=std_dev) - norm.cdf(value1, loc=mean, scale=std_dev))
        else:
            return jsonify({'error': 'Tipo de cálculo inválido'}), 400

        x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 500)
        y = norm.pdf(x, loc=mean, scale=std_dev)

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='Distribuição Normal')
        if calculation_type == 'above':
            plt.fill_between(x, y, where=(x > value), color='blue', alpha=0.3, label='Área')
        elif calculation_type == 'below':
            plt.fill_between(x, y, where=(x < value), color='blue', alpha=0.3, label='Área')
        elif calculation_type == 'between':
            plt.fill_between(x, y, where=(x > value1) & (x < value2), color='blue', alpha=0.3, label='Área')
        elif calculation_type == 'outside':
            plt.fill_between(x, y, where=(x < value1) | (x > value2), color='blue', alpha=0.3, label='Área')

        plt.title('Distribuição Normal')
        plt.xlabel('X')
        plt.ylabel('Densidade')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        return jsonify({'probability': probability, 'plot': plot_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/calculate/bayes', methods=['POST'])
def calculate_bayes():
    try:
        data = request.json
        n = data['n']
        probabilities = data['probabilities']

        p_b = sum(p['P(Ai)'] * p['P(B|Ai)'] for p in probabilities)

        p_a_given_b = []
        for i, p in enumerate(probabilities, start=1):
            p_ai_given_b = (p['P(Ai)'] * p['P(B|Ai)']) / p_b
            details = (
                f"P(A{i}|B) = (P(A{i}) * P(B|A{i})) / P(B) = "
                f"({p['P(Ai)']:.4f} * {p['P(B|Ai)']:.4f}) / {p_b:.4f} = {p_ai_given_b:.4f}"
            )
            p_a_given_b.append({'P(Ai|B)': p_ai_given_b, 'details': details})

        return jsonify({'P(B)': p_b, 'P(Ai|B)': p_a_given_b})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Arquivo não selecionado.'}), 400

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        cor_grafico = request.form.get('cor', 'aqua')

        try:
            df_csv = pd.read_csv(filepath)
            col_name = df_csv.columns[0]
            data = pd.to_numeric(df_csv[col_name], errors='coerce').dropna()

            if data.empty:
                return jsonify({'error': 'O arquivo deve conter dados numéricos na primeira coluna.'}), 400

            n = float(data.count())
            
            # ATENÇÃO: Voltando à fórmula exata original com log natural (base e)
            k = int(1 + 3.33 * np.log(n))

            minimo = float(data.min())
            maximo = float(data.max())
            At = float(maximo - minimo)
            h = float(At / k)

            limites = [minimo + i * h for i in range(k + 1)]

            classes = pd.cut(data, bins=limites, right=False, include_lowest=True)
            f_i = classes.value_counts().sort_index().values

            tab = pd.DataFrame()
            tab['LI'] = limites[:-1]
            tab['LS'] = limites[1:]
            tab['classes'] = [f"[{li:.2f} |- {ls:.2f})" for li, ls in zip(tab['LI'], tab['LS'])]
            
            tab['f_i'] = f_i
            tab['f_ri'] = tab['f_i'] / n

            tab['FAC'] = tab['f_ri'].cumsum()
            tab['FAD'] = tab['f_ri'][::-1].cumsum()[::-1]

            tab['X_Mi'] = (tab['LI'] + tab['LS']) / 2
            tab['X_Mi*f_ri'] = tab['X_Mi'] * tab['f_ri']

            media = float(tab['X_Mi*f_ri'].sum())
            tab['((X_Mi - media)^2)*f_i'] = ((tab['X_Mi'] - media) ** 2) * tab['f_i']

            if n > 1:
                variancia = float(tab['((X_Mi - media)^2)*f_i'].sum()) / (n - 1)
            else:
                variancia = 0.0
            
            desvio_padrao = float(variancia ** 0.5)

            max_fi = tab['f_i'].max()
            modas = tab[tab['f_i'] == max_fi]['X_Mi'].tolist()
            if len(modas) == 1:
                moda_str = f"{modas[0]:.4f}"
            else:
                moda_str = "{" + ", ".join([f"{m:.4f}" for m in modas]) + "}"

            try:
                classe_mediana_idx = tab[tab['FAC'] >= 0.5].index[0]
                LI_md = float(tab.loc[classe_mediana_idx, 'LI'])
                fi_md = float(tab.loc[classe_mediana_idx, 'f_i'])
                
                if classe_mediana_idx == 0:
                    Fant = 0.0
                else:
                    Fant = float(tab.loc[:classe_mediana_idx - 1, 'f_i'].sum())

                if fi_md > 0:
                    mediana = float(LI_md + (((n / 2) - Fant) / fi_md) * h)
                else:
                    mediana = 0.0
            except IndexError:
                mediana = 0.0

            # Gerando a tabela de frequencias para excel com a fonte obrigatoria
            tab_excel = tab.copy()
            tab_excel.loc[len(tab_excel)] = ["Fonte: elaborado pelo autor usando https://jonathanestatistica.com.br"] + [""] * (len(tab_excel.columns) - 1)
            excel_filename = 'tabela_frequencias.xlsx'
            excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)
            tab_excel.to_excel(excel_path, index=False)

            # Gerando a tabela de medidas de resumo para excel com a fonte obrigatoria
            stats_dict = {
                'n': n,
                'k': k,
                'Máximo': maximo,
                'Mínimo': minimo,
                'Amplitude (At)': At,
                'h (Largura da Classe)': h,
                'Média': media,
                'Mediana': mediana,
                'Moda': moda_str,
                'Variância Amostral': variancia,
                'Desvio Padrão': desvio_padrao
            }
            
            df_metrics = pd.DataFrame(list(stats_dict.items()), columns=['Medida', 'Valor'])
            df_metrics.loc[len(df_metrics)] = ["Fonte: elaborado pelo autor usando https://jonathanestatistica.com.br", ""]
            metrics_filename = 'medidas_resumo.xlsx'
            metrics_path = os.path.join(app.config['UPLOAD_FOLDER'], metrics_filename)
            df_metrics.to_excel(metrics_path, index=False)

            # Histograma ajustado sem as classes fantasmas
            plt.figure(figsize=(10, 6))
            plt.bar(
                x=tab['LI'],
                height=tab['f_ri'],
                width=h,
                align='edge',
                color=cor_grafico,
                edgecolor='black',
                linewidth=1.2
            )

            limites_x = tab['LI'].tolist() + [tab['LS'].iloc[-1]]
            plt.title(f'Histograma das Frequências de {col_name}')
            plt.xlabel(col_name)
            plt.ylabel('Frequência Relativa')
            plt.xticks(limites_x, rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.subplots_adjust(bottom=0.25)
            plt.figtext(0.125, 0.02, "Fonte: elaborado pelo autor usando https://jonathanestatistica.com.br", wrap=True, horizontalalignment='left', fontsize=10)

            img_filename = 'histograma.png'
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            plt.savefig(img_path, dpi=300)
            plt.close()

            tab = tab.replace({np.nan: None})
            tab_json = tab.to_dict(orient='records')

            return jsonify({
                'col_name': col_name,
                'table': tab_json,
                'stats': {
                    'n': n,
                    'k': k,
                    'maximo': maximo,
                    'minimo': minimo,
                    'amplitude': At,
                    'h': h,
                    'media': media,
                    'mediana': mediana,
                    'moda': moda_str,
                    'variancia': variancia,
                    'desvio_padrao': desvio_padrao
                },
                'img_url': f'/download_file/{img_filename}',
                'excel_url': f'/download_file/{excel_filename}',
                'metrics_url': f'/download_file/{metrics_filename}'
            })

        except Exception as e:
            return jsonify({'error': f'Erro ao processar o arquivo: {str(e)}'}), 500

    else:
        return jsonify({'error': 'Apenas arquivos .csv são permitidos.'}), 400

@app.route('/download_file/<filename>', methods=['GET'])
def download_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Erro ao baixar o arquivo: {str(e)}'}), 500

@app.route('/calculate/ci_mean', methods=['POST'])
def calculate_ci_mean():
    try:
        data = request.json
        sample_mean = float(data['sample_mean'])
        alpha = float(data['alpha'])
        std_dev = float(data['std_dev'])
        sample_size = int(data['sample_size'])

        z_score = norm.ppf(1 - alpha / 2)
        margin_error = z_score * (std_dev / np.sqrt(sample_size))
        lower_limit = sample_mean - margin_error
        upper_limit = sample_mean + margin_error

        x_min = sample_mean - 4 * (std_dev / np.sqrt(sample_size))
        x_max = sample_mean + 4 * (std_dev / np.sqrt(sample_size))
        x = np.linspace(x_min, x_max, 1000)
        y = norm.pdf(x, loc=sample_mean, scale=std_dev / np.sqrt(sample_size))

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='Distribuição Normal', color='blue')
        plt.fill_between(x, y, where=(x >= lower_limit) & (x <= upper_limit), color='green', alpha=0.3, label='Área de Confiança (95%)')
        plt.axvline(sample_mean, color='red', linestyle='--', label='Média Amostral')
        plt.axvline(lower_limit, color='orange', linestyle='--', label='Limite Inferior')
        plt.axvline(upper_limit, color='orange', linestyle='--', label='Limite Superior')
        plt.title('Intervalo de Confiança para a Média')
        plt.xlabel('Valores')
        plt.ylabel('Densidade')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        result = f"Com {(1 - alpha) * 100:.1f}% de confiança, a média está entre {lower_limit:.4f} e {upper_limit:.4f}, com base na amostra."
        return jsonify({'result': result, 'plot': plot_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/calculate/ci_proportion', methods=['POST'])
def calculate_ci_proportion():
    try:
        data = request.json
        sample_proportion = float(data['sample_proportion'])
        alpha = float(data['alpha'])
        population_proportion = float(data['population_proportion'])
        sample_size = int(data['sample_size'])

        z_score = norm.ppf(1 - alpha / 2)
        variance = population_proportion * (1 - population_proportion) / sample_size
        margin_error = z_score * np.sqrt(variance)
        lower_limit = sample_proportion - margin_error
        upper_limit = sample_proportion + margin_error

        x = np.linspace(sample_proportion - 4 * np.sqrt(variance), sample_proportion + 4 * np.sqrt(variance), 500)
        y = norm.pdf(x, loc=sample_proportion, scale=np.sqrt(variance))

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='Distribuição Normal')
        plt.fill_between(x, y, where=(x >= lower_limit) & (x <= upper_limit), color='blue', alpha=0.3, label='Área de Confiança')
        plt.axvline(sample_proportion, color='red', linestyle='--', label='Proporção Amostral')
        plt.axvline(lower_limit, color='green', linestyle='--', label='Limite Inferior')
        plt.axvline(upper_limit, color='green', linestyle='--', label='Limite Superior')
        plt.title('Intervalo de Confiança para a Proporção')
        plt.xlabel('X')
        plt.ylabel('Densidade')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        result = f"Com {(1 - alpha) * 100:.1f}% de confiança, a proporção está entre {lower_limit:.4f} e {upper_limit:.4f}, com base na amostra."
        return jsonify({'result': result, 'plot': plot_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/calculate/hypothesis_mean', methods=['POST'])
def calculate_hypothesis_mean():
    try:
        data = request.json

        required_fields = ['test_type', 'h0', 'alpha', 'sample_size', 'sample_mean', 'std_dev']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"O campo '{field}' é obrigatório."}), 400

        test_type = data['test_type']
        h0 = float(data['h0'])
        alpha = float(data['alpha'])
        sample_size = int(data['sample_size'])
        sample_mean = float(data['sample_mean'])
        std_dev = float(data['std_dev'])

        z_score = (sample_mean - h0) / (std_dev / np.sqrt(sample_size))

        if test_type == "bilateral":
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        elif test_type == "left":
            p_value = norm.cdf(z_score)
        else:
            p_value = 1 - norm.cdf(z_score)

        reject_h0 = p_value < alpha

        conclusion = (
            f"A probabilidade de errar é {p_value:.4f} e como ela é baixa (< {alpha}), "
            f"rejeita-se H₀ com {100 * (1 - alpha):.2f}% de confiança."
            if reject_h0
            else f"A probabilidade de errar é {p_value:.4f}, que não é baixa (>= {alpha}). "
                 f"Portanto, não se rejeita H₀ com {100 * (1 - alpha):.2f}% de confiança."
        )

        result = f"Z Amostral (ẑ): {z_score:.4f}, P-valor: {p_value:.4f}. {conclusion}"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

@app.route('/calculate/hypothesis_proportion', methods=['POST'])
def calculate_hypothesis_proportion():
    try:
        data = request.json

        required_fields = ['test_type', 'h0', 'alpha', 'sample_size', 'sample_proportion']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"O campo '{field}' é obrigatório."}), 400

        test_type = data['test_type']
        h0 = float(data['h0'])
        alpha = float(data['alpha'])
        sample_size = int(data['sample_size'])
        sample_proportion = float(data['sample_proportion'])

        std_error = np.sqrt(h0 * (1 - h0) / sample_size)
        z_score = (sample_proportion - h0) / std_error

        if test_type == "bilateral":
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        elif test_type == "left":
            p_value = norm.cdf(z_score)
        else:
            p_value = 1 - norm.cdf(z_score)

        reject_h0 = p_value < alpha

        conclusion = (
            f"A probabilidade de errar é {p_value:.4f} e como ela é baixa (< {alpha}), "
            f"rejeita-se H₀ com {100 * (1 - alpha):.2f}% de confiança."
            if reject_h0
            else f"A probabilidade de errar é {p_value:.4f}, que não é baixa (>= {alpha}). "
                 f"Portanto, não se rejeita H₀ com {100 * (1 - alpha):.2f}% de confiança."
        )

        result = f"Z Amostral (ẑ): {z_score:.4f}, P-valor: {p_value:.4f}. {conclusion}"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500
    
@app.route('/calculate/hypothesis_mean_unknown_variance', methods=['POST'])
def calculate_hypothesis_mean_unknown_variance():
    try:
        data = request.json

        required_fields = ['test_type', 'h0', 'alpha', 'sample_size', 'sample_mean', 'sample_variance']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"O campo '{field}' é obrigatório."}), 400

        test_type = data['test_type']
        h0 = float(data['h0'])
        alpha = float(data['alpha'])
        sample_size = int(data['sample_size'])
        sample_mean = float(data['sample_mean'])
        sample_variance = float(data['sample_variance'])

        if not (0 < alpha < 1):
            return jsonify({"error": "O nível de significância (alpha) deve estar entre 0 e 1."}), 400

        std_error = np.sqrt(sample_variance / sample_size)
        t_score = (sample_mean - h0) / std_error
        degrees_of_freedom = sample_size - 1

        if test_type == "bilateral":
            p_value = 2 * (1 - t.cdf(abs(t_score), df=degrees_of_freedom))
        elif test_type == "left":
            p_value = t.cdf(t_score, df=degrees_of_freedom)
        else:
            p_value = 1 - t.cdf(t_score, df=degrees_of_freedom)

        reject_h0 = p_value < alpha

        conclusion = (
            f"A probabilidade de errar é {p_value:.4f} e como ela é baixa (< {alpha}), "
            f"rejeita-se H₀ com {100 * (1 - alpha):.2f}% de confiança."
            if reject_h0
            else f"A probabilidade de errar é {p_value:.4f}, que não é baixa (>= {alpha}). "
                 f"Portanto, não se rejeita H₀ com {100 * (1 - alpha):.2f}% de confiança."
        )

        result = f"T Amostral (t̂): {t_score:.4f}, P-valor: {p_value:.4f}. {conclusion}"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

@app.route('/calculate/difference_means', methods=['POST'])
def calculate_difference_means():
    import traceback

    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "Arquivo não enviado."}), 400

        filename = file.filename.lower()
        if filename == '':
            return jsonify({"error": "Nenhum arquivo selecionado."}), 400

        test_type = request.form.get('test_type', 'bilateral')
        independent = request.form.get('independent') == 'on'
        alpha = float(request.form.get('alpha', 0.05))

        if filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return jsonify({"error": "Formato inválido. Use CSV ou XLSX."}), 400

        if df.shape[1] != 2:
            return jsonify({"error": "O arquivo deve ter exatamente 2 colunas."}), 400

        col1, col2 = df.columns

        data1 = pd.to_numeric(df[col1], errors='coerce').dropna().reset_index(drop=True)
        data2 = pd.to_numeric(df[col2], errors='coerce').dropna().reset_index(drop=True)

        if len(data1) < 2 or len(data2) < 2:
            return jsonify({"error": "Dados insuficientes."}), 400

        if independent:
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        else:
            if len(data1) != len(data2):
                return jsonify({"error": "Para teste pareado, os vetores devem ter o mesmo tamanho."}), 400
            t_stat, p_value = stats.ttest_rel(data1, data2)

        if test_type == 'left':
            p_value = p_value / 2 if t_stat < 0 else 1 - (p_value / 2)
        elif test_type == 'right':
            p_value = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)

        reject = p_value < alpha

        conclusion = (
            f"p-valor = {p_value:.4f} → rejeita-se H0"
            if reject else
            f"p-valor = {p_value:.4f} → não se rejeita H0"
        )

        result_data = {
            "z_score": float(t_stat),
            "p_value": float(p_value),
            "conclusion": conclusion
        }

        output = io.BytesIO()
        wb = openpyxl.Workbook()
        ws = wb.active

        ws.append(["Teste de Diferença de Médias"])
        ws.append([])
        ws.append(["t-stat", t_stat])
        ws.append(["p-valor", p_value])
        ws.append(["Conclusão", conclusion])

        wb.save(output)
        output.seek(0)
        app.config['last_result'] = output.read()

        return jsonify({
            "result": result_data,
            "download_url": "/download/difference_means"
        })

    except Exception as e:
        print("\n===== ERRO =====")
        traceback.print_exc()
        print("================\n")
        return jsonify({"error": str(e)}), 500

@app.route('/download/difference_means', methods=['GET'])
def download_difference_means():
    try:
        result_file = app.config.get('last_result')
        if not result_file:
            return jsonify({"error": "Nenhum resultado encontrado para download."}), 400

        return send_file(
            io.BytesIO(result_file),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='difference_means_results.xlsx'
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro ao processar o arquivo: {str(e)}'}), 500

##################################################
# ================================
# 🔁 Carregamento e Processamento
# ================================
df_path = os.path.join("data", "brasileirao_serieA_2025_completo.csv")

if not os.path.exists(df_path):
    print("🔁 Arquivo CSV da Série A não encontrado. Baixando...")
    try:
        from scraper_brasileirao_2025 import baixar_dados_brasileirao
        baixar_dados_brasileirao()
        print("✅ Dados da Série A baixados com sucesso.")
    except Exception as e:
        raise FileNotFoundError("❌ Falha ao baixar os dados automaticamente: " + str(e))

df = pd.read_csv(df_path)

if "Data e Hora" in df.columns and not np.issubdtype(df["Data e Hora"].dtype, np.datetime64):
    df["Data e Hora"] = pd.to_datetime(df["Data e Hora"])

df_casa = df.copy()
if "Home Team" in df_casa.columns:
    df_casa["Time"] = df_casa["Home Team"]
else:
    df_casa["Time"] = df_casa["Time Casa"]

df_fora = df.copy()
if "Away Team" in df_fora.columns:
    df_fora["Time"] = df_fora["Away Team"]
else:
    df_fora["Time"] = df_fora["Time Visitante"]

df = pd.concat([df_casa, df_fora], ignore_index=True)

def classificar_resultado(row, time):
    if row["Time Casa"] == time:
        saldo = row["Gols Casa"] - row["Gols Visitante"]
    elif row["Time Visitante"] == time:
        saldo = row["Gols Visitante"] - row["Gols Casa"]
    else:
        return "Outro"
    if saldo > 0:
        return "Vitória"
    elif saldo < 0:
        return "Derrota"
    else:
        return "Empate"

def calcular_gols_sofridos(row, time):
    if row["Time Casa"] == time:
        return row["Gols Visitante"]
    elif row["Time Visitante"] == time:
        return row["Gols Casa"]
    else:
        return 0

df_original = df.copy()

@app.route('/calculate/regressao', methods=['POST'])
def calcular_regressao():
    if 'arquivo' not in request.files or request.files['arquivo'].filename == '':
        return jsonify({'erro': 'Nenhum arquivo enviado.'})

    arquivo = request.files['arquivo']
    df = pd.read_csv(arquivo)
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    resumo = model.summary().as_text()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0, 1, resumo, fontsize=8, va='top', family='monospace')
    ax.axis('off')
    jpg_path = os.path.join('static', 'regressao_summary.jpg')
    fig.savefig(jpg_path, bbox_inches='tight')
    plt.close(fig)

    xlsx_path = os.path.join('static', 'regressao_resultados.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Dados')
        pd.DataFrame({'Resumo': [resumo]}).to_excel(writer, index=False, sheet_name='Resumo')

    return jsonify({
        'summary_jpg': '/' + jpg_path,
        'result_xlsx': '/' + xlsx_path
    })

@app.route("/calculate/difference_proportions", methods=["POST"])
def calculate_difference_proportions():
    try:
        from scipy.stats import norm
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        import base64

        data = request.get_json()
        p1_hat = data["p1_hat"]
        p2_hat = data["p2_hat"]
        n1 = data["n1"]
        n2 = data["n2"]
        alpha = data["alpha"]
        test_type = data["test_type"]

        p_pool = ((p1_hat * n1) + (p2_hat * n2)) / (n1 + n2)
        q_pool = 1 - p_pool

        se = (p_pool * q_pool * (1/n1 + 1/n2)) ** 0.5
        z_score = (p1_hat - p2_hat) / se

        if test_type == "bilateral":
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        elif test_type == "left":
            p_value = norm.cdf(z_score)
        else: 
            p_value = 1 - norm.cdf(z_score)

        if p_value < alpha:
            conclusion = "Rejeita-se H₀: há evidência de diferença entre as proporções."
        else:
            conclusion = "Não se rejeita H₀: não há evidência de diferença entre as proporções."

        x = np.linspace(-4, 4, 1000)
        y = norm.pdf(x)
        fig, ax = plt.subplots()
        ax.plot(x, y, label='Distribuição N(0,1)')

        ax.axvline(z_score, color='red', linestyle='--', label=f'Z = {z_score:.2f}')

        if test_type == "bilateral":
            z_alpha = norm.ppf(1 - alpha / 2)
            ax.fill_between(x, y, where=(x < -z_alpha) | (x > z_alpha), color='orange', alpha=0.3)
            ax.axvline(-z_alpha, color='orange', linestyle='--', label=f'-Zα/2 = {-z_alpha:.2f}')
            ax.axvline(z_alpha, color='orange', linestyle='--', label=f'Zα/2 = {z_alpha:.2f}')
        elif test_type == "left":
            z_alpha = norm.ppf(alpha)
            ax.fill_between(x, y, where=(x < z_alpha), color='orange', alpha=0.3)
            ax.axvline(z_alpha, color='orange', linestyle='--', label=f'Zα = {z_alpha:.2f}')
        elif test_type == "right":
            z_alpha = norm.ppf(1 - alpha)
            ax.fill_between(x, y, where=(x > z_alpha), color='orange', alpha=0.3)
            ax.axvline(z_alpha, color='orange', linestyle='--', label=f'Zα = {z_alpha:.2f}')

        ax.set_title("Distribuição N(0,1) com Região Crítica")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return jsonify({
            "z_score": z_score,
            "p_value": p_value,
            "conclusion": conclusion,
            "plot": plot_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/estatisticasdofutebol")
def estatisticas_futebol_home():
    return render_template("ciga.html")

@app.route("/estatisticas2025")
def estatisticas_2025():
    df = pd.read_csv("data/brasileirao_serieA_2025_completo.csv")

    def classificar_resultado(row):
        if row["Time Referência"] == row["Time Casa"]:
            if row["Gols Casa"] > row["Gols Visitante"]:
                return "Vitória"
            elif row["Gols Casa"] < row["Gols Visitante"]:
                return "Derrota"
            else:
                return "Empate"
        elif row["Time Referência"] == row["Time Visitante"]:
            if row["Gols Visitante"] > row["Gols Casa"]:
                return "Vitória"
            elif row["Gols Visitante"] < row["Gols Casa"]:
                return "Derrota"
            else:
                return "Empate"
        else:
            return "Indefinido"

    df["Resultado"] = df.apply(classificar_resultado, axis=1)

    campeonatos = sorted(df["Campeonato"].dropna().unique())
    times = sorted(df["Time Referência"].dropna().unique())

    return render_template(
        "estatisticas_2025.html",
        campeonatos=campeonatos,
        times=times,
        dados_jogos=df.to_dict(orient='records')
    )

@app.route("/mercadojogadores")
def mercado_jogadores():
    return render_template("mercado_jogadores.html")

@app.route("/analisabase")
def analise_base():
    return render_template("analise_base.html")

@app.route("/atualizar-dados", methods=["POST"])
def atualizar_dados():
    thread = threading.Thread(target=scraper.rodar_scraper_com_progresso)
    thread.start()
    return jsonify({"mensagem": "⏳ Scraping iniciado. Isso pode levar alguns minutos. Acompanhe a barra de progresso abaixo."})

@app.route("/progresso", methods=["GET"])
def progresso_status():
    return jsonify({"progresso": scraper.progresso_atual})

if __name__ == '__main__':
    app.run(debug=True)