from flask import Flask, render_template

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

if __name__ == "__main__":
    app.run(debug=True)

