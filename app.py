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

        # Calcular a probabilidade com base no tipo
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

        # Gerar gráfico
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

        # Salvar gráfico como base64
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

        # Calculando P(B) usando o teorema da probabilidade total
        p_b = sum(p['P(Ai)'] * p['P(B|Ai)'] for p in probabilities)

        # Calculando P(Ai|B) para cada partição
        p_a_given_b = []
        for i, p in enumerate(probabilities, start=1):
            p_ai_given_b = (p['P(Ai)'] * p['P(B|Ai)']) / p_b
            details = (
                f"P(A{i}|B) = (P(A{i}) * P(B|A{i})) / P(B) = "
                f"({p['P(Ai)']:.4f} * {p['P(B|Ai)']:.4f}) / {p_b:.4f} = {p_ai_given_b:.4f}"
            )
            p_a_given_b.append({'P(Ai|B)': p_ai_given_b, 'details': details})

        # Retornando resultados
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

        try:
            data = pd.read_csv(filepath, header=None).iloc[:, 0]
            if data.empty or not np.issubdtype(data.dtype, np.number):
                return jsonify({'error': 'O arquivo deve conter apenas dados numéricos em uma coluna.'}), 400

            n = len(data)
            k = ceil(sqrt(n))
            a_t = data.max() - data.min()

            decimal_places = max(len(str(x).split('.')[-1]) if '.' in str(x) else 0 for x in data)
            h = round(a_t / k, decimal_places)

            bins = np.arange(data.min(), data.max() + h, h)
            if bins[-1] < data.max():
                bins = np.append(bins, bins[-1] + h)

            freq, intervals = np.histogram(data, bins=bins)
            intervals = [(round(intervals[i], decimal_places), round(intervals[i + 1], decimal_places)) for i in range(len(intervals) - 1)]
            midpoints = [(interval[0] + interval[1]) / 2 for interval in intervals]
            relative_freq = freq / n
            fac_vals = np.cumsum(relative_freq)
            fad_vals = np.cumsum(relative_freq[::-1])[::-1]

            fr_decimal = [f"{r:.4f}" for r in relative_freq]
            fac = [f"{v:.4f}" for v in fac_vals]
            fad = [f"{v:.4f}" for v in fad_vals]

            xmi_times_fr = [round(m * r, 4) for m, r in zip(midpoints, relative_freq)]
            mean = sum(xmi_times_fr)  # soma(xmi * fr) é a média

            mid_rounded = [round(m, 2) for m in midpoints]
            fr_div_xmi = [round(r / m, 4) if m != 0 else 0 for m, r in zip(midpoints, relative_freq)]
            xmi_pow_fr = [round(m ** r, 4) if m > 0 else 0 for m, r in zip(midpoints, relative_freq)]
            variance_terms = [round(((m - mean) ** 2) * f, 4) for m, f in zip(midpoints, freq)]
            mean_dev_terms = [round(abs(m - mean) * f, 4) for m, f in zip(midpoints, freq)]

            total_fi = sum(freq)
            total_fr = f"{sum(relative_freq):.4f}"
            total_xmi_times_fr = round(sum(xmi_times_fr), 4)
            total_fr_div_xmi = round(sum(fr_div_xmi), 4)
            prod_xmi_pow_fr = round(np.prod(xmi_pow_fr), 4)
            total_var_term = round(sum(variance_terms), 4)
            total_dev_term = round(sum(mean_dev_terms), 4)

            data_rows = list(zip(
                [f"{i[0]} - {i[1]}" for i in intervals], freq, fr_decimal, fac, fad,
                mid_rounded, xmi_times_fr, fr_div_xmi, xmi_pow_fr, variance_terms, mean_dev_terms
            ))

            data_rows.append([
                "Total", total_fi, total_fr, "-", "-", "-", total_xmi_times_fr, total_fr_div_xmi,
                prod_xmi_pow_fr, total_var_term, total_dev_term
            ])

            columns = [
                "Intervalo", "$f_i$", "$fr_i$", "$FAC(fr)$", "$FAD(fr)$",
                "$\\overline{x}_{M_i}$", "$\\overline{x}_{M_i} \\cdot fr_i$",
                "$fr_i / \\overline{x}_{M_i}$", "$\\overline{x}_{M_i}^{fr_i}$",
                "$((\\overline{x}_{M_i} - \\overline{x})^2) \\cdot f_i$",
                "$|\\overline{x}_{M_i} - \\overline{x}| \\cdot f_i$"
            ]

            df = pd.DataFrame(data_rows, columns=columns)

            fig, ax = plt.subplots(figsize=(18, 5))
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.6)

            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.get_text().set_weight('bold')
                if col == 0 or col == len(columns) - 1:
                    cell.visible_edges = 'horizontal'

            plt.subplots_adjust(bottom=0.15)
            plt.figtext(0.125, 0.02, "Fonte: elaborado pelo autor usando https://jonathanestatistica.com.br",
                        wrap=True, horizontalalignment='left', fontsize=10)

            img_path = os.path.join("static", "uploads", 'tabela_frequencia.jpg')
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()

            return jsonify({"img_url": f"/{img_path}"})

        except Exception as e:
            return jsonify({'error': f'Erro ao processar o arquivo: {str(e)}'}), 500

    else:
        return jsonify({'error': 'Apenas arquivos .csv são permitidos.'}), 400

    
@app.route('/calculate/ci_mean', methods=['POST'])
def calculate_ci_mean():
    try:
        data = request.json
        sample_mean = float(data['sample_mean'])
        alpha = float(data['alpha'])
        std_dev = float(data['std_dev'])
        sample_size = int(data['sample_size'])

        # Cálculo do z-score
        z_score = norm.ppf(1 - alpha / 2)

        # Cálculo do intervalo de confiança
        margin_error = z_score * (std_dev / np.sqrt(sample_size))
        lower_limit = sample_mean - margin_error
        upper_limit = sample_mean + margin_error

        # Ajustar o intervalo do eixo x para focar na área do intervalo de confiança
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

        # Salvar gráfico como base64
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

        # Cálculo do z-score
        z_score = norm.ppf(1 - alpha / 2)

        # Variância da proporção
        variance = population_proportion * (1 - population_proportion) / sample_size

        # Cálculo do intervalo de confiança
        margin_error = z_score * np.sqrt(variance)
        lower_limit = sample_proportion - margin_error
        upper_limit = sample_proportion + margin_error

        # Gerar gráfico
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

        # Salvar gráfico como base64
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

        # Verificar campos obrigatórios
        required_fields = ['test_type', 'h0', 'alpha', 'sample_size', 'sample_mean', 'sample_variance']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"O campo '{field}' é obrigatório."}), 400

        # Obter os parâmetros
        test_type = data['test_type']
        h0 = float(data['h0'])
        alpha = float(data['alpha'])
        sample_size = int(data['sample_size'])
        sample_mean = float(data['sample_mean'])
        sample_variance = float(data['sample_variance'])

        # Verificar validade de alpha
        if not (0 < alpha < 1):
            return jsonify({"error": "O nível de significância (alpha) deve estar entre 0 e 1."}), 400

        # Calcular t-score
        std_error = np.sqrt(sample_variance / sample_size)
        t_score = (sample_mean - h0) / std_error
        degrees_of_freedom = sample_size - 1

        # Calcular p-valor
        if test_type == "bilateral":
            p_value = 2 * (1 - t.cdf(abs(t_score), df=degrees_of_freedom))
        elif test_type == "left":
            p_value = t.cdf(t_score, df=degrees_of_freedom)
        else:  # "right"
            p_value = 1 - t.cdf(t_score, df=degrees_of_freedom)

        # Decisão
        reject_h0 = p_value < alpha

        # Gerar conclusão
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
    try:
        file = request.files['file']
        test_type = request.form['test_type']
        independent = request.form['independent'] == 'true'
        alpha = float(request.form['alpha'])

        df = pd.read_excel(file)
        if len(df.columns) != 2:
            return jsonify({"error": "O arquivo deve conter exatamente duas colunas."}), 400

        col1, col2 = df.columns
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()

        if independent:
            t_stat, p_value = stats.ttest_ind(data1, data2)
        else:
            t_stat, p_value = stats.ttest_rel(data1, data2)

        # Ajustar p-valor para testes unilaterais
        if test_type == 'left':
            p_value = p_value / 2 if t_stat < 0 else 1 - (p_value / 2)
        elif test_type == 'right':
            p_value = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)

        reject_h0 = p_value < alpha

        # Definir hipóteses e conclusão
        if test_type == 'bilateral':
            h0 = f"H₀: μ₁ = μ₂"
            h1 = f"H₁: μ₁ ≠ μ₂"
            explanation_h1 = f"média de {col1} é diferente da média de {col2}"
        elif test_type == 'left':
            h0 = f"H₀: μ₁ - μ₂ ≥ 0"
            h1 = f"H₁: μ₁ - μ₂ < 0"
            explanation_h1 = f"média de {col1} é menor que a média de {col2}"
        elif test_type == 'right':
            h0 = f"H₀: μ₁ - μ₂ ≤ 0"
            h1 = f"H₁: μ₁ - μ₂ > 0"
            explanation_h1 = f"média de {col1} é maior que a média de {col2}"

        conclusion = (
            f"Como o p-valor = {p_value:.4f}, rejeita-se H₀. Logo, há evidência para aceitar a hipótese alternativa."
            if reject_h0
            else f"Como o p-valor = {p_value:.4f}, não se rejeita H₀. Logo, não há evidência suficiente para aceitar a hipótese alternativa."
        )

        result_data = {
            "z_score": t_stat,
            "p_value": p_value,
            "conclusion": conclusion,
            "col1_name": col1,
            "col2_name": col2
        }

        # Salvar resultados no Excel
        output = io.BytesIO()
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Resultados"

        # Escrever cabeçalho
        sheet.append(["Teste de diferença de médias"])
        sheet.append([])

        # Z-score e p-valor
        sheet.append(["Z-score", "p-valor"])
        sheet.append([t_stat, p_value])
        sheet.append([])

        # Hipóteses
        sheet.append(["Hipótese Nula", "Explicação"])
        sheet.append([h0, f"média de {col1} é igual à média de {col2}"])
        sheet.append([])
        sheet.append(["Hipótese Alternativa", "Explicação"])
        sheet.append([h1, explanation_h1])
        sheet.append([])

        # Conclusão
        sheet.append(["Conclusão:"])
        sheet.append([conclusion])

        workbook.save(output)
        output.seek(0)
        app.config['last_result'] = output.read()

        return jsonify({"result": result_data, "download_url": "/download/difference_means"})

    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

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

# Verifica e baixa os dados automaticamente se necessário
if not os.path.exists(df_path):
    print("🔁 Arquivo CSV da Série A não encontrado. Baixando...")
    try:
        from scraper_brasileirao_2025 import baixar_dados_brasileirao
        baixar_dados_brasileirao()
        print("✅ Dados da Série A baixados com sucesso.")
    except Exception as e:
        raise FileNotFoundError("❌ Falha ao baixar os dados automaticamente: " + str(e))

# Carregamento seguro dos dados
df = pd.read_csv(df_path)

# Converte a coluna de data
if "Data e Hora" in df.columns and not np.issubdtype(df["Data e Hora"].dtype, np.datetime64):
    df["Data e Hora"] = pd.to_datetime(df["Data e Hora"])

# Cria coluna 'Time' a partir de jogos em casa e fora (modelo multilinha)
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

# Classifica resultado por time principal
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

# Adiciona função para calcular gols sofridos
def calcular_gols_sofridos(row, time):
    if row["Time Casa"] == time:
        return row["Gols Visitante"]
    elif row["Time Visitante"] == time:
        return row["Gols Casa"]
    else:
        return 0

# 🔁 Geração de colunas auxiliares e resumo por time (será usada nos gráficos)
# ➤ Essa lógica pode ser usada em callbacks para atualizar os gráficos
# ➤ Exemplos de visualizações a gerar:
# - Frequência de Vitórias, Empates, Derrotas
# - Gols marcados e sofridos por jogo
# - Cartões amarelos e vermelhos por partida

# Armazena o DataFrame original para uso em callbacks
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

    # Salvar como imagem
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0, 1, resumo, fontsize=8, va='top', family='monospace')
    ax.axis('off')
    jpg_path = os.path.join('static', 'regressao_summary.jpg')
    fig.savefig(jpg_path, bbox_inches='tight')
    plt.close(fig)

    # Salvar como Excel
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

        # Proporção combinada sob H0
        p_pool = ((p1_hat * n1) + (p2_hat * n2)) / (n1 + n2)
        q_pool = 1 - p_pool

        # Estatística de teste Z
        se = (p_pool * q_pool * (1/n1 + 1/n2)) ** 0.5
        z_score = (p1_hat - p2_hat) / se

        # P-valor
        if test_type == "bilateral":
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        elif test_type == "left":
            p_value = norm.cdf(z_score)
        else:  # "right"
            p_value = 1 - norm.cdf(z_score)

        # Conclusão
        if p_value < alpha:
            conclusion = "Rejeita-se H₀: há evidência de diferença entre as proporções."
        else:
            conclusion = "Não se rejeita H₀: não há evidência de diferença entre as proporções."

        # Geração do gráfico com região crítica adaptada
        x = np.linspace(-4, 4, 1000)
        y = norm.pdf(x)
        fig, ax = plt.subplots()
        ax.plot(x, y, label='Distribuição N(0,1)')

        # Linha do valor observado de Z
        ax.axvline(z_score, color='red', linestyle='--', label=f'Z = {z_score:.2f}')

        # Região crítica e linhas limiares
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


# ROTA: Página com os 3 cards principais (Estatísticas do Futebol)
@app.route("/estatisticasdofutebol")
def estatisticas_futebol_home():
    return render_template("ciga.html")

@app.route("/estatisticas2025")
def estatisticas_2025():
    df = pd.read_csv("data/brasileirao_serieA_2025_completo.csv")

    # Criar coluna 'Resultado' com base no time referência
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


# ROTA: Página Mercado de Jogadores
@app.route("/mercadojogadores")
def mercado_jogadores():
    return render_template("mercado_jogadores.html")

# ROTA: Página Análise da Base Sub-20
@app.route("/analisabase")
def analise_base():
    return render_template("analise_base.html")

# ROTA: Inicia o scraping com barra de progresso
@app.route("/atualizar-dados", methods=["POST"])
def atualizar_dados():
    thread = threading.Thread(target=scraper.rodar_scraper_com_progresso)
    thread.start()
    return jsonify({"mensagem": "⏳ Scraping iniciado. Isso pode levar alguns minutos. Acompanhe a barra de progresso abaixo."})

# ROTA: Retorna progresso atual
@app.route("/progresso", methods=["GET"])
def progresso_status():
    return jsonify({"progresso": scraper.progresso_atual})


if __name__ == '__main__':
    app.run(debug=True)
