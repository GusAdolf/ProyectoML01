#IMPORTAR LIBRERIA PARA USAR FRAMEWORK FLASK
from flask import Flask
from flask import render_template
import os
from flask import request
import backend
##llamado a flask
app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'IMG')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER



##servicio web
#Carga de IMAGENES 

@app.route('/', methods = ["GET","POST"])
def home():
    return render_template('home.html')
def frase():
    return render_template('frase.html')

#Documento
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)
        racismo = backend.cargaListas('static/FILES/racismo.txt')
        idenGenero = backend.cargaListas('static\FILES\idenGen.txt')
        clase = backend.cargaListas('static\FILES\clase.txt')
        edad = 'asd'
        d1 = backend.cargaDoc(f.filename)
        """Construcción de Colección con listas discriminatorias"""
        colecGeneral = backend.colecCompleta(d1,racismo,idenGenero,clase,edad)
        """JACCARD"""
        Jaccard = backend.jaccardCompleto(colecGeneral)
        """COSENO"""
        Coseno = backend.cosenoVect(colecGeneral)
        return render_template("carga.html", name = f.filename,matrix1 = Jaccard, coseno = Coseno)

#Colección
@app.route('/success2', methods = ['POST'])  
def success2():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)
        racismo = backend.cargaListas('static/FILES/racismo.txt')
        idenGenero = backend.cargaListas('static\FILES\idenGen.txt')
        clase = backend.cargaListas('static\FILES\clase.txt')
        edad = 'asd'
        d1 = backend.cargaColec(f.filename)
        """Construcción de Colección con listas discriminatorias"""
        colecGeneral = backend.colecCompleta(d1,racismo,idenGenero,clase,edad)
        """JACCARD"""
        Jaccard = backend.jaccardCompleto(colecGeneral)
        """COSENO"""
        Coseno = backend.cosenoVect(colecGeneral)
        return render_template("carga.html", name = f.filename,matrix1 = Jaccard, coseno = Coseno)

@app.route('/info')
def info():
    
    cami = os.path.join(app.config['UPLOAD_FOLDER'], 'cami.jpg')
    gus=os.path.join(app.config['UPLOAD_FOLDER'], 'gus.jpg')
    joss=os.path.join(app.config['UPLOAD_FOLDER'], 'joss.jpg')
    jona=os.path.join(app.config['UPLOAD_FOLDER'], 'jona.jpg')
    alexis=os.path.join(app.config['UPLOAD_FOLDER'], 'alexis.jpeg')
    return render_template("info.html", c=cami,g=gus,j=joss,jo=jona,a=alexis)
       

@app.route('/resultados', methods = ['POST'])
def resultados():
    if request.method == 'POST':  
        user_input = request.args.get('user_input')
        racismo = backend.cargaListas('static/FILES/racismo.txt')
        idenGenero = backend.cargaListas('static\FILES\idenGen.txt')
        clase = backend.cargaListas('static\FILES\clase.txt')
        edad = 'asd'
        d1 = str(user_input)
        """Construcción de Colección con listas discriminatorias"""
        colecGeneral = [d1,racismo,idenGenero,clase,edad]
        """JACCARD"""
        Jaccard = backend.jaccardCompleto(colecGeneral)
        """COSENO"""
        Coseno = backend.cosenoVect(colecGeneral)
        return render_template('resultados.html',result=user_input,matrix1 = Jaccard, coseno = Coseno)



##ejecutar el servicio web
if __name__=='__main__':
    #OJO QUITAR EL DEBUG EN PRODUCCION
    app.run(host='0.0.0.0', port=5000, debug=True)