import matplotlib
matplotlib.use('Agg')  # Backend não interativo para gráficos
from flask import Flask, render_template, request, jsonify, send_file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import os
from math import comb, exp, factorial, ceil, sqrt
from werkzeug.utils import secure_filename

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

# Upload and Process File for Frequency Distribution
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

            # Cálculos de distribuição de frequências
            n = len(data)
            k = ceil(sqrt(n))
            a_t = data.max() - data.min()
            h = round(a_t / k, len(str(data.iloc[0]).split('.')[-1]))
            bins = np.arange(data.min(), data.max() + h, h)
            freq, intervals = np.histogram(data, bins=bins)

            intervals = [(round(intervals[i], 2), round(intervals[i + 1], 2)) for i in range(len(intervals) - 1)]
            midpoints = [(interval[0] + interval[1]) / 2 for interval in intervals]

            relative_freq = freq / n
            fac = np.cumsum(freq)
            fad = np.cumsum(freq[::-1])[::-1]

            # Criar tabela como JSON
            table = {
                "Intervalo": [f"{interval[0]} - {interval[1]}" for interval in intervals],
                "fi": freq.tolist(),
                "fr": relative_freq.tolist(),
                "FAC": fac.tolist(),
                "FAD": fad.tolist(),
            }

            return jsonify({"table": table})
        except Exception as e:
            return jsonify({'error': f'Erro ao processar o arquivo: {str(e)}'}), 500

    else:
        return jsonify({'error': 'Apenas arquivos .csv são permitidos.'}), 400
    
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        data = pd.read_csv(file, header=None).squeeze()

        if data.empty:
            return jsonify({'error': 'O arquivo está vazio.'})

        # Processamento exemplo
        n = len(data)
        k = int(np.ceil(np.sqrt(n)))
        amplitude_total = data.max() - data.min()
        h = round(amplitude_total / k, 2)

        return jsonify({
            'message': 'Arquivo processado com sucesso!',
            'Amplitude Total': amplitude_total,
            'Tamanho da Classe': h,
            'k': k
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "__main__":
    app.run(debug=True)