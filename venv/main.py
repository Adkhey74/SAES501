from flask import Flask,render_template,send_file

app = Flask(__name__)

@app.route('/')
def home():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('home.html',active_page='Home')

@app.route('/graph.html')
def graph():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('graph.html',active_page='Graph')


@app.route('/tableau.html')
def tableau():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('tableau.html',active_page='Tableau')

@app.route('/detection_inconfort.html')
def detection():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('detection_inconfort.html',active_page='Detection')

@app.route('/connexion_compte.html')
def connexion():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('connexion_compte.html',active_page='Connexion')
@app.route('/configuration.html')
def configuration():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('configuration.html',active_page='Configuration')








if __name__ == '__main__':
    app.run(debug=True)
