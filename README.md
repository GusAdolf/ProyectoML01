# ProyectoML01
TEMA: Sistema basado en contenidos que permita analizar el grado y tipo de discriminación de un contenido textual
INTEGRANTES:
1. Camila Cedeño
2. Joselyn Cherrez
3. Gustavo Contreras
4. Jonthan Guerra
5. Alexis Vallejo.

REQUERIMIENTOS:
1. El sistema debe ser capaz de reconocer si un mensaje tiene algún grado de
discriminación y el tipo de discriminación. Por lo tanto, ante una nueva entrada
textual, el sistema debe indicar si el mensaje tiene algún grado de discriminación
o no. Si existe discriminación se debe especificar el nivel porcentual de
discriminación (comparar la cantidad de tokens discriminatorios con respecto al
número total de tokens). El sistema debe basarse en contenidos para el idioma
español - dialecto (contexto) ecuatoriano y el mínimo de clases de discriminación
es de cuatro. Las clases pueden ser seleccionadas desde las determinadas por la
ONG Amnistía Internacional: https://www.amnesty.org/es/what-wedo/discrimination/
2. Cuando el sistema detecta que existe algún grado de discriminación, además,
debe indicar el tipo de discriminación. Si existe más de un tipo, el sistema debe
devolver los porcentajes de discriminación de cada clase.
3. El sistema debe realizar el proceso de NLP, tanto en los diccionarios fijos, como
en las nuevas entradas textuales y para medir la similitud se deben utilizar al
menos dos métricas (Jaccard y Coseno Vectorial). Por lo tanto, el sistema debe
mostrar los grados y tipos de discriminación con las dos metodologías.
4. Crear un sistema que se debe desplegar en la web (no en local) y permita ingresar
un texto y analizar el nivel y tipo de discriminación. El sistema debe incluir la
opción de carga de ficheros externos, para el análisis de un texto externo.
5. Representar en diagramas de bloques, pseudocódigos o flujogramas sus procesos
(sugerencia usar Visual Paradigm)
