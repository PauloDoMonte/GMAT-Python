import time, os
import pandas as pd
import numpy as np
import pickle, subprocess, sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score

def normalizar_dados():
	"""
	Normaliza os dados de treinamento usando StandardScaler.

	Esta função lê três arquivos CSV ("dados.txt", "dados1.txt" e "dados2.txt") e os combina em um único DataFrame. Em seguida, separa os recursos (X) e
	as variáveis-alvo (Y) com base em índices de coluna especificados. O StandardScaler é usado para normalizar X e Y para um melhor desempenho de treinamento da rede neural.

	Retorna:
		tupla: Uma tupla contendo os dados de treinamento normalizados (X_norm, Y_norm),
			   os escaladores para X e Y (escala_X, escala_Y) e o número total de pontos de dados.

	Levanta:
		FileNotFoundError: Se algum dos arquivos CSV não for encontrado.
		Exception: Se ocorrer algum erro inesperado ao ler os arquivos CSV.
	"""

	try:
		antes = time.time()
		print("Carregando o Dataset")
		df1 = pd.read_csv("dados.txt")
		df2 = pd.read_csv("dados1.txt")
		df3 = pd.read_csv("dados2.txt")
		df = pd.concat([df1,df2,df3])
		print(f"Dataset Carregado com {len(df)} dados\tDemorou: {time.time()-antes} segundos")

	except FileNotFoundError as e:
		print(f"Erro: Arquivo não encontrado - {e.filename}")
	except Exception as e:
		print(f"Erro inesperado ao ler os arquivos txt: {e}")

	
	X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,28]].values
	Y = df.iloc[:,[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]].values
	#Y =  df.iloc[:,[24,25,26,27]].values

	#X = df.iloc[:, [0,1,2,3,4,5]].values
	#Y = df.iloc[:,[12,13,14,15,16,17,24,25,26,27]].values

	escala_X = StandardScaler()
	escala_Y = StandardScaler()

	escala_X.fit(X)
	escala_Y.fit(Y)

	X_norm = escala_X.transform(X)
	Y_norm = escala_Y.transform(Y)

	return X_norm, Y_norm,escala_X,escala_Y

def criar_rede_neural(X_norm,Y_norm):
	"""
	Cria e treina uma rede neural regressora Multi-Layer Perceptron (MLP).

	Esta função define a arquitetura da rede neural com várias camadas ocultas
	e funções de ativação específicas ('relu'), solver ('adam') e parâmetros de
	taxa de aprendizado. A função tenta carregar uma rede previamente treinada de
	'rna.pkl' se ela existir. Caso contrário, treina uma nova rede usando os dados
	de treinamento normalizados fornecidos (X_norm, Y_norm). A rede é dividida em
	conjuntos de treinamento e teste usando uma divisão de 75/25.

	Retorna:
		tupla: Uma tupla contendo a rede neural treinada (rna), os conjuntos de
			   treinamento e teste (x_train, y_train, x_test, y_test).
	"""
	
	print("Iniciando uma nova RNA")
	rna = MLPRegressor(hidden_layer_sizes=(13,65,65,65,65,65,16,), activation='relu', solver='adam', alpha=1, batch_size='auto', 
	learning_rate='constant', learning_rate_init=1e-4, max_iter=10000, warm_start=True,
	tol=1e-8, verbose=2, early_stopping=True,n_iter_no_change=100)

	if os.path.exists('rna.pkl'):

		print("Carregando a RNA")		
		with open('rna.pkl', mode='rb') as f:
			rna = pickle.load(f)
			print("RNA Carregado\n")

	else: print("Não foi possivel carregar a RNA treinada")

	x_train, x_test, y_train, y_test = train_test_split(X_norm, Y_norm, test_size=0.25)

	rna.fit(x_train, y_train)	

	return(rna,x_train, y_train,x_test,y_test)

def treinar():

	print("Inciando o treinamento")
	
	X_norm, Y_norm, escala_X,escala_Y= normalizar_dados()

	rna,x_train, y_train,x_test,y_test= criar_rede_neural(X_norm,Y_norm)

	# Calcula as métricas no conjunto de teste
	previsao = rna.predict(x_test)
	r2 = r2_score(y_test, previsao)
	mse = mean_squared_error(y_test, previsao)
	mae = mean_absolute_error(y_test, previsao)
	rmse = root_mean_squared_error(y_test, previsao)
	explained_variance = explained_variance_score(y_test, previsao)
	median_ae = median_absolute_error(y_test, previsao)

	print(f"\nR2:\t{r2}\nMSE:\t{mse}\nMAE:\t{mae}\nRMSE:\t{rmse}")
	print(f"\nVariancia Explicada: {explained_variance}")
	print(f"Erro Médio Absoluto Mediano: {median_ae}")

	metricas = [r2,mse,mae,rmse,explained_variance,median_ae]

	with open('rna.pkl', mode='wb') as f:
		pickle.dump((rna),f)

	# Salva o modelo treinado
	arquivo = (f"Cerebro/cerebro_len:({len(x_train)},{len(x_test)})_r2:{r2}_.pkl")
	with open(arquivo, mode='wb') as f:
		pickle.dump((rna, metricas, escala_X, escala_Y, len(x_train), len(x_test)), f)

	previsao = escala_Y.inverse_transform(previsao)
	y_test = escala_Y.inverse_transform(y_test)

	indice, v_previsao,v_test = [],[],[]
	vx_test,vx_previsao = [],[]

	for i in range(100):
		indice.append(i)

		vx_previsao.append(previsao[i][12])
		vx_test.append(y_test[i][12])

		v_previsao.append(np.sqrt((previsao[i][14]*previsao[i][14])+(previsao[i][12]*previsao[i][12])+(previsao[i][13]*previsao[i][13])))
		v_test.append(np.sqrt((y_test[i][14]*y_test[i][14])+(y_test[i][12]*y_test[i][12])+(y_test[i][13]*y_test[i][13])))

	import matplotlib.pyplot as plt

	plt.scatter(indice,vx_previsao,color='red',s=5,label='Previsto')
	plt.scatter(indice,vx_test,color='green',s=5,label='Real')

	plt.grid()
	plt.legend()
	plt.show()

	plt.scatter(indice,v_previsao,color='red',s=5,label='Previsto')
	plt.scatter(indice,v_test,color='green',s=5,label='Real')

	plt.grid()
	plt.legend()
	plt.show()

def testar():
	import matplotlib.pyplot as plt

	r2_, treino_ = [], []

	orbita_inicial 	= [[10353.176327976927,0.07207178536086266,38.22197756708569,112.67176268288793,297.41443832960306,336.79468665957523, 15114.16691162969,0.1658790491131772,47.93007613917435,73.07574719532745,282.4851941744873,21.310690476010148,1e-6]]

	diretorio_atual = f"{os.getcwd()}/Cerebro/"
	arquivos = os.listdir(diretorio_atual)
	arquivos_pkl = [arquivo for arquivo in arquivos if arquivo.startswith("cerebro") and arquivo.endswith(".pkl")]
	for arquivo in arquivos_pkl:
		arquivo = f"Cerebro/{arquivo}"
		print(arquivo)
		with open(arquivo, 'rb') as f:
			rna, metricas, best_score, best_params_,escala_x, escala_y, treino, teste = pickle.load(f)	
			r2 = metricas[0]
			mse = metricas[1]
			mae = metricas[2]
			rmse = metricas[3]
			explained_variance = metricas[4]
			median_ae = metricas[5]

			r2_.append(r2)
			treino_.append(treino+teste)

			x = orbita_inicial
			escala_x.fit(x)
			x_norm = escala_x.transform(x)
			previsao = escala_y.inverse_transform(rna.predict(x_norm))
			nf_sma, nf_ecc, nf_inc, nf_raan, nf_aop, nf_ta,	df_sma, df_ecc, df_inc, df_raan, df_aop, df_ta, vx, vy, vz,t = np.split(previsao[0], 16, axis=0)

	print(treino_)
	plt.scatter(treino_,r2_,s=7,c='red')
	plt.legend()
	plt.grid()
	plt.title("Evolução do R2 em função da quantidade de dados")
	plt.ylabel("R2")
	plt.xlabel("Quantidade de dados")
	plt.show()

if(sys.argv[1] == 'treinar'):

	treinar()

elif (sys.argv[1] == 'testar'):testar()

else:
	print("Comando desconhecido")
