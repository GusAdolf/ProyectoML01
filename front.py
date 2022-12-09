#IMPORTAR LIBRERIA PARA USAR FRAMEWORK FLASK
from flask import Flask
from flask import render_template
import os
from flask import request
import backend
import numpy as np
##llamado a flask
app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'IMG')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER



##servicio web
@app.route('/', methods = ["GET","POST"])
def home():
    return render_template('home.html')

#Documento
@app.route('/success', methods = ["GET","POST"])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        doc = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
        racismo = backend.cargaListas('static/FILES/racismo.txt')
        idenGenero = backend.cargaListas('static\FILES\idenGen.txt')
        clase = backend.cargaListas('static\FILES\clase.txt')
        edad = backend.cargaListas('static\FILES\edad.txt')
        d1 = backend.cargaDoc(doc)
        n = 1
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
        items,flag = backend.cargaDoc(doc)
        print(items)
        jacRes1 = (jacRes[0,:])
        sorRes1 = (sorRes[0,:])
        cosRes1 = (cosRes[0,:])
        return render_template("docu.html",doc = items,jaccard = jacRes1, coseno = cosRes1, sorensen = sorRes1)

#Colección
@app.route('/success2', methods = ["GET","POST"])  
def success2():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        doc = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
        racismo = backend.cargaListas('static/FILES/racismo.txt')
        idenGenero = backend.cargaListas('static\FILES\idenGen.txt')
        clase = backend.cargaListas('static\FILES\clase.txt')
        edad = backend.cargaListas('static\FILES\edad.txt')
        d1 = backend.cargaColec(doc)
        n = len(d1[0])
        """Construcción de Colección con listas discriminatorias"""
        colecGeneral = backend.colecCompleta(d1,racismo,idenGenero,clase,edad)
        """JACCARD"""
        Jaccard = backend.jaccardCompleto(colecGeneral,1,2,0)
        jacRes = np.round(backend.respuesta(Jaccard,n),2)
        """Sorensen"""
        Sorensen = backend.sorensenCompleto(colecGeneral,1,2,0)
        sorRes = np.round(backend.respuesta(Sorensen,n),2)
        """COSENO"""
        Coseno = backend.cosenoVectN(colecGeneral,1,2,0)
        cosRes = np.round(backend.respuesta(Coseno,n),2)
        items,flag = backend.cargaColec(doc)
        [x.encode('utf-8').decode('utf-8') for x in items]
        print(items)
        return render_template("coleccion.html",len=n,doc = items,jaccard = jacRes, coseno = cosRes, sorensen = sorRes)

#Imagen
@app.route('/success3', methods = ["GET","POST"])  
def success3():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        img = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
        racismo = backend.cargaListas('static/FILES/racismo.txt')
        idenGenero = backend.cargaListas('static\FILES\idenGen.txt')
        clase = backend.cargaListas('static\FILES\clase.txt')
        edad = backend.cargaListas('static\FILES\edad.txt')
        d1 = backend.cargaDoc(img)
        n = 1
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
        items,flag = backend.cargaDoc(img)
        print(items)
        jacRes1 = (jacRes[0,:])
        sorRes1 = (sorRes[0,:])
        cosRes1 = (cosRes[0,:])
        return render_template("img.html",len=n,doc = items,jaccard = jacRes1, coseno = cosRes1, sorensen = sorRes1, img = img)


@app.route('/info')
def info():
    
    cami = os.path.join(app.config['UPLOAD_FOLDER'], 'cami.jpg')
    gus=os.path.join(app.config['UPLOAD_FOLDER'], 'gus.jpg')
    joss=os.path.join(app.config['UPLOAD_FOLDER'], 'joss.jpg')
    jona=os.path.join(app.config['UPLOAD_FOLDER'], 'jona.jpg')
    alexis=os.path.join(app.config['UPLOAD_FOLDER'], 'alexis.jpeg')
    return render_template("info.html", c=cami,g=gus,j=joss,jo=jona,a=alexis)
       
#Frase
@app.route('/submit', methods=["GET","POST"])
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
    jacRes1 = (jacRes[0,:])
    sorRes1 = (sorRes[0,:])
    cosRes1 = (cosRes[0,:])
    return render_template('resultados.html',result=request.form['text'],jaccard = jacRes1, coseno = cosRes1, sorensen = sorRes1)


##ejecutar el servicio web
if __name__=='__main__':
    #OJO QUITAR EL DEBUG EN PRODUCCION
    app.run(host='0.0.0.0', port=5000, debug=True)