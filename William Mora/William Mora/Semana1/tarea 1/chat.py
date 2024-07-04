import pandas as pd
#pip install -U scikit-learn scipy matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
 # Para convertir el texto a vectores
from scipy import spatial 
# Para medir la distancia entre vectores

#data = pd.read_csv('C:/Users/William Wallace/Desktop/carpeta visual/tarea 1/Data.csv',sep='[;]',engine='python')
# Si no se cuenta con ella, se simula tener una
data = pd.DataFrame([['horarios de entrenamiento','lunes a viernes de 6 pm a 8 pm'],['tipos de artes marciales','mma hapkido muay thai jiu jitsu'],['costo o valor','130000']])
data.columns = ['question','answer']
#print(data.columns)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data.question) 
# La línea de código anterior convierte la columna 'question' en vectores,
# es decir, cada pregunta la convierte en una serie de números.
print (pd.DataFrame(vectorizer.get_feature_names_out())) 
#ver valores de las preguntas
#print (pd.DataFrame(X.toarray()[0]))
#ver vectorizado

question = ['tipos de clases']
#Lo que hará el bot para responder esa pregunta, es simplemente buscar a cuál de las preguntas se parece más. Para ello, primero deberá vectorizar la pregunta:
vectorized_question = vectorizer.transform(question).toarray()
#Lo que haremos a continuación, es identificar cuál es la pregunta de la base de datos, que más se parezca a la pregunta que le hicimos al bot.
tree = spatial.KDTree(X.toarray())
# 1 fila La distancia que existe entre la pregunta realizada y la pregunta de la base de datos más parecida.
# 2 La respuesta de la base de datos más parecida.
print (pd.DataFrame(tree.query(vectorized_question)))

#la pregunta a la cual se asocio
print (data.iloc[tree.query(vectorized_question)[1][0]].question)
#la respuseta
print (data.iloc[tree.query(vectorized_question)[1][0]].answer)