from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
import pandas as pd
import pickle

def treinar():
    df = pd.read_csv("rede_neural.txt")

    X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]].values
    Y = df.iloc[:,[12,13,14]].values

    escala_X = StandardScaler()
    escala_Y = StandardScaler()

    escala_X.fit(X)
    escala_Y.fit(Y)

    X_norm = escala_X.transform(X)
    Y_norm = escala_Y.transform(Y)

    # Define os parâmetros para otimização

    learning = 1e-4
    no_change = 100
    layer=(12,32,16,16,3)

    param_grid = {
        'hidden_layer_sizes': [layer],
        'alpha': [1],
        'learning_rate_init': [learning],
        'n_iter_no_change': [no_change],
        'tol': [1e-18],
        'solver': ['adam'],
        'activation': ['relu'],    }

    # Cria o objeto GridSearchCV
    rna = MLPRegressor(max_iter=10000,learning_rate="constant", verbose=2, early_stopping=True)

    cv=5

    grid_search = GridSearchCV(rna, param_grid, cv=cv, scoring='r2')

    # Ajusta o objeto GridSearchCV aos dados
    grid_search.fit(X_norm, Y_norm)

    # Treina o modelo final com os melhores parâmetros encontrados
    rna = grid_search.best_estimator_
    x_train, x_test, y_train, y_test = train_test_split(X_norm, Y_norm, test_size=0.25)

    rna.fit(x_train, y_train)

    # Calcula as métricas no conjunto de teste
    previsao = rna.predict(x_test)
    r2 = r2_score(y_test, previsao)
    mse = mean_squared_error(y_test, previsao)
    mae = mean_absolute_error(y_test, previsao)
    rmse = root_mean_squared_error(y_test, previsao)
    explained_variance = explained_variance_score(y_test, previsao)
    median_ae = median_absolute_error(y_test, previsao)

    # Acessa os melhores parâmetros encontrados
    print(f"Melhores parametros:{grid_search.best_params_}")

    # Acessa o melhor escore alcançado
    best_score = grid_search.best_score_
    print("R2 [Cross-Validation] :", best_score)

    print(f"\nR2:\t{r2}\nMSE:\t{mse}\nMAE:\t{mae}\nRMSE:\t{rmse}")
    print(f"\nVariancia Explicada: {explained_variance}")
    print(f"Erro Médio Absoluto Mediano: {median_ae}")

    metricas = [r2,mse,mae,rmse,explained_variance,median_ae]

    # Salva o modelo treinado
    arquivo = (f"cerebro_learning:{learning}_nochange:{no_change}_cv:{cv}_layer:{layer}.pkl")
    with open(arquivo, mode='wb') as f:
        pickle.dump((rna, metricas, best_score, grid_search.best_params_, escala_X, escala_Y, len(x_train), len(x_test)), f)

def testar():
	import os
	import numpy as np

	orbita_inicial 	= [[10353.176327976927,0.07207178536086266,38.22197756708569,112.67176268288793,297.41443832960306,336.79468665957523, 15114.16691162969,0.1658790491131772,47.93007613917435,73.07574719532745,282.4851941744873,21.310690476010148]]

	with open('estatistica.csv', mode='w') as s:
		s.write("R2_Cross_Validation;R2;MAE;Arquivo\n")

	diretorio_atual = os.getcwd()
	arquivos = os.listdir(diretorio_atual)
	arquivos_pkl = [arquivo for arquivo in arquivos if arquivo.startswith("cerebro") and arquivo.endswith(".pkl")]

	for arquivo in arquivos_pkl:
		
		with open(arquivo, 'rb') as f:
			rna, metricas, best_score, best_params_,escala_x, escala_y, treino, teste = pickle.load(f)	
			r2 = metricas[0]
			mse = metricas[1]
			mae = metricas[2]
			rmse = metricas[3]
			explained_variance = metricas[4]
			median_ae = metricas[5]

			if(r2 >= 0.5):
				if(best_score >= 0.4):
					x = orbita_inicial
					escala_x.fit(x)
					x_norm = escala_x.transform(x)

					previsao = rna.predict(x_norm)
					vx, vy, vz = np.split(previsao[0], 3, axis=0)

					#with open('estatistica.csv', mode='a') as s:
					#	s.write(f"{best_score};{r2};{mae};{arquivo}\n")



					print(f"R2 [cross-validation]:{round(best_score,6)}\tR2:{round(r2,6)}\tMAE:{round(mae,6)}km/s\t{arquivo}")
					print(f"Vx:{round(vx[0],4)} (km/s)\tVy:{round(vy[0],4)} (km/s)\tVz:{round(vz[0],4)} (km/s)")
					print(f"Treinei:{treino}\tTestei:{teste}\n")

treinar()
#testar()