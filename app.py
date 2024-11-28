from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")  # Serves templates/index.html

# Dashboard route
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")  # Serves templates/dashboard.html

@app.route("/projeto")
def projeto():
    return render_template("projeto.html")

@app.route("/codigos")
def codigo():
    return render_template("codigos.html")

@app.route("/videos")
def videos():
    return render_template("videos.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# Ensino routes
@app.route("/ensino")
def ensino():
    return render_template("ensino.html")

@app.route("/ensino/biologia")
def biologia():
    return render_template("ensino/biologia.html")

@app.route("/ensino/biologia/plano_de_ensino")
def biologia_plano_de_ensino():
    return render_template("ensino/biologia/plano_de_ensino.html")

@app.route("/ensino/biologia/listas")
def biologia_listas():
    return render_template("ensino/biologia/listas.html")

@app.route("/ensino/biologia/exercicios_resolvidos")
def biologia_exercicios_resolvidos():
    return render_template("ensino/biologia/exercicios_resolvidos.html")

@app.route("/ensino/fisioterapia")
def fisioterapia():
    return render_template("ensino/fisioterapia.html")

@app.route("/ensino/fisioterapia/plano_de_ensino")
def fisioterapia_plano_de_ensino():
    return render_template("ensino/fisioterapia/plano_de_ensino.html")

@app.route("/ensino/fisioterapia/listas")
def fisioterapia_listas():
    return render_template("ensino/fisioterapia/listas.html")

@app.route("/ensino/fisioterapia/exercicios_resolvidos")
def fisioterapia_exercicios_resolvidos():
    return render_template("ensino/fisioterapia/exercicios_resolvidos.html")

@app.route("/ensino/terapia_ocupacional")
def terapia_ocupacional():
    return render_template("ensino/terapia_ocupacional.html")

@app.route("/ensino/terapia_ocupacional/plano_de_ensino")
def terapia_ocupacional_plano_de_ensino():
    return render_template("ensino/terapia_ocupacional/plano_de_ensino.html")

@app.route("/ensino/terapia_ocupacional/listas")
def terapia_ocupacional_listas():
    return render_template("ensino/terapia_ocupacional/listas.html")

@app.route("/ensino/terapia_ocupacional/exercicios_resolvidos")
def terapia_ocupacional_exercicios_resolvidos():
    return render_template("ensino/terapia_ocupacional/exercicios_resolvidos.html")

@app.route("/ensino/biomedicina_farmacia")
def biomedicina_farmacia():
    return render_template("ensino/biomedicina_farmacia.html")

@app.route("/ensino/biomedicina_farmacia/plano_de_ensino")
def biomedicina_farmacia_plano_de_ensino():
    return render_template("ensino/biomedicina_farmacia/plano_de_ensino.html")

@app.route("/ensino/biomedicina_farmacia/listas")
def biomedicina_farmacia_listas():
    return render_template("ensino/biomedicina_farmacia/listas.html")

@app.route("/ensino/biomedicina_farmacia/exercicios_resolvidos")
def biomedicina_farmacia_exercicios_resolvidos():
    return render_template("ensino/biomedicina_farmacia/exercicios_resolvidos.html")

# Rota para servir PDFs na pasta static/pdf
@app.route('/plano_de_ensino/pdf/<filename>')
def plano_pdf(filename):
    return send_from_directory('static/pdf', filename)

if __name__ == "__main__":
    app.run(debug=True)