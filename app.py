from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Ensino main route
@app.route("/ensino/")
def ensino():
    return render_template("ensino.html")

# Discipline routes
@app.route("/ensino/<disciplina>/")
def disciplina(disciplina):
    return render_template("disciplina.html", disciplina=disciplina)

# Download Plano de Ensino
@app.route("/ensino/<disciplina>/plano_de_ensino/")
def download_plano_de_ensino(disciplina):
    file_path = f"static/ensino/{disciplina}"
    file_name = f"plano_ensino_{disciplina}.pdf"
    try:
        print(f"Tentando acessar o arquivo: {file_path}/{file_name}")
        return send_from_directory(file_path, file_name, as_attachment=True)
    except FileNotFoundError:
        return f"Erro: Arquivo '{file_name}' não encontrado na disciplina '{disciplina}'.", 404
    except Exception as e:
        return f"Erro ao tentar baixar o arquivo: {e}", 500

# Listas route
@app.route("/ensino/<disciplina>/listas/")
def listas(disciplina):
    lista_files = [f"lista{i}.pdf" for i in range(1, 6)]
    return render_template("listas.html", disciplina=disciplina, lista_files=lista_files)

# Exercícios Resolvidos route
@app.route("/ensino/<disciplina>/exercicios_resolvidos/")
def exercicios_resolvidos(disciplina):
    exercises_dir = f"static/ensino/{disciplina}/exercicios_resolvidos"
    files = os.listdir(exercises_dir) if os.path.exists(exercises_dir) else []
    support_file = "suporte_lista_5.xlsx"  # Arquivo compartilhado
    return render_template("exercicios_resolvidos.html", disciplina=disciplina, files=files, support_file=support_file)

# Serve static files for Listas
@app.route("/static/listas/<filename>")
def download_lista(filename):
    return send_from_directory("static/listas", filename, as_attachment=True)

# Serve static files for Exercícios Resolvidos
@app.route("/static/ensino/<disciplina>/exercicios_resolvidos/<filename>")
def download_exercise(disciplina, filename):
    return send_from_directory(f"static/ensino/{disciplina}/exercicios_resolvidos", filename, as_attachment=True)

# Additional Pages
@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/dashboard/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/projeto/")
def projeto():
    return render_template("projeto.html")

@app.route("/codigos/")
def codigos():
    return render_template("codigos.html")

@app.route("/videos/")
def videos():
    return render_template("videos.html")

@app.route("/contact/")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)