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
        edad = backend.cargaListas('static\FILES\edad.txt')
        d1 = backend.cargaDoc(f.filename)
        n = len(d1[0])
        """Construcción de Colección con listas discriminatorias"""
        colecGeneral = backend.colecCompleta(d1,racismo,idenGenero,clase,edad)
        """JACCARD"""
        Jaccard = backend.jaccardCompleto(colecGeneral,1,2,0)
        jacRes = backend.respuesta(Jaccard,n)
        """Sorensen"""
        Sorensen = backend.sorensenCompleto(colecGeneral,1,2,0)
        sorRes = backend.respuesta(Sorensen,n)
        """COSENO"""
        Coseno = backend.cosenoVectN(colecGeneral,1,2,0)
        cosRes = backend.respuesta(Coseno,n)
        return render_template("carga.html", name = f.filename,jaccard = jacRes, coseno = cosRes, sorensen = sorRes)

#Colección
@app.route('/success2', methods = ['POST'])  
def success2():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)
        racismo = backend.cargaListas('static/FILES/racismo.txt')
        idenGenero = backend.cargaListas('static\FILES\idenGen.txt')
        clase = backend.cargaListas('static\FILES\clase.txt')
        edad = backend.cargaListas('static\FILES\edad.txt')
        d1 = backend.cargaColec(f.filename)
        n = len(d1[0])
        """Construcción de Colección con listas discriminatorias"""
        colecGeneral = backend.colecCompleta(d1,racismo,idenGenero,clase,edad)
        """JACCARD"""
        Jaccard = backend.jaccardCompleto(colecGeneral,1,2,0)
        jacRes = backend.respuesta(Jaccard,n)
        """Sorensen"""
        Sorensen = backend.sorensenCompleto(colecGeneral,1,2,0)
        sorRes = backend.respuesta(Sorensen,n)
        """COSENO"""
        Coseno = backend.cosenoVectN(colecGeneral,1,2,0)
        cosRes = backend.respuesta(Coseno,n)
        return render_template("carga.html", name = f.filename,jaccard = jacRes, coseno = cosRes, sorensen = sorRes)

@app.route('/info')
def info():
    
    cami = os.path.join(app.config['UPLOAD_FOLDER'], 'cami.jpg')
    gus=os.path.join(app.config['UPLOAD_FOLDER'], 'gus.jpg')
    joss=os.path.join(app.config['UPLOAD_FOLDER'], 'joss.jpg')
    jona=os.path.join(app.config['UPLOAD_FOLDER'], 'jona.jpg')
    alexis=os.path.join(app.config['UPLOAD_FOLDER'], 'alexis.jpeg')
    return render_template("info.html", c=cami,g=gus,j=joss,jo=jona,a=alexis)
       
   
@app.route('/submit', methods=['POST'])
def submit():
    racismo = backend.cargaListas('static/FILES/racismo.txt')
    idenGenero = backend.cargaListas('static\FILES\idenGen.txt')
    clase = backend.cargaListas('static\FILES\clase.txt')
    edad = backend.cargaListas('static\FILES\edad.txt')
    d1 = (request.form['text'],1)
    n = 1
    #Construcción de Colección con listas discriminatorias
    colecGeneral = backend.colecCompleta(d1,racismo,idenGenero,clase,edad)
    #JACCARD
    Jaccard = backend.jaccardCompleto(colecGeneral,1,2,0)
    jacRes = backend.respuesta(Jaccard,n)
    #Sorensen
    Sorensen = backend.sorensenCompleto(colecGeneral,1,2,0)
    sorRes = backend.respuesta(Sorensen,n)
    #COSENO
    Coseno = backend.cosenoVectN(colecGeneral,1,2,0)
    cosRes = backend.respuesta(Coseno,n)
    return render_template('resultados.html',result=request.form['text'],jaccard = jacRes, coseno = cosRes, sorensen = sorRes)


##ejecutar el servicio web
if __name__=='__main__':
    #OJO QUITAR EL DEBUG EN PRODUCCION
    app.run(host='0.0.0.0', port=5000, debug=True)