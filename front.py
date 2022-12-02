#IMPORTAR LIBRERIA PARA USAR FRAMEWORK FLASK
from flask import Flask
from flask import render_template
import os
from flask import request
##llamado a flask
app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'IMG')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER



##servicio web
#Carga de IMAGENES 

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/info')
def info():
    
    cami = os.path.join(app.config['UPLOAD_FOLDER'], 'cami.jpg')
    gus=os.path.join(app.config['UPLOAD_FOLDER'], 'gus.jpg')
    joss=os.path.join(app.config['UPLOAD_FOLDER'], 'joss.jpg')
    jona=os.path.join(app.config['UPLOAD_FOLDER'], 'jona.jpg')
    alexis=os.path.join(app.config['UPLOAD_FOLDER'], 'alexis.jpeg')
    return render_template("info.html", c=cami,g=gus,j=joss,jo=jona,a=alexis)
       

@app.route('/resultados')
def resultados():
    user_input = request.args.get('user_input')
    return render_template('resultados.html',result=user_input)



##ejecutar el servicio web
if __name__=='__main__':
    #OJO QUITAR EL DEBUG EN PRODUCCION
    app.run(host='0.0.0.0', port=5000, debug=True)