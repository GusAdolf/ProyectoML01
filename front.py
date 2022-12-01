#IMPORTAR LIBRERIA PARA USAR FRAMEWORK FLASK
from flask import Flask
from flask import render_template

##llamado a flask
app = Flask(__name__)

##servicio web
@app.route('/')
def home():
    return render_template('home.html')



@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/carga')
def carga():
    return render_template('carga.html')


##ejecutar el servicio web
if __name__=='__main__':
    #OJO QUITAR EL DEBUG EN PRODUCCION
    app.run(host='0.0.0.0', port=5000, debug=True)