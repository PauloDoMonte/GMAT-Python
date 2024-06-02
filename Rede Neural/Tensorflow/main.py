import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, subprocess, sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='logs/colisao')

modelos = [(26,52,26,52,26,52)]
m = 0


class CurrentEpochCallback(tf.keras.callbacks.Callback):
	def on_epoch_begin(self, epoch, logs=None):
		print(f"\nEpoch {epoch + 1} em andamento...")

def normalizar_dados():

	try:
		antes = time.time()
		print("Carregando o Dataset")
		df1 = pd.read_csv("dados.txt")
		df2 = pd.read_csv("dados1.txt")
		df3 = pd.read_csv("dados2.txt")
		df4 = pd.read_csv("dados3.txt")
		df = pd.concat([df1,df2,df3,df4])
		print(f"Dataset Carregado com {len(df)} dados\tDemorou: {time.time()-antes} segundos")
# 1845,93 dados/MB
# Para ter 1 milhão = 541,72 MB | 10 Milhões = 5,41 GB

	except FileNotFoundError as e:
		print(f"Erro: Arquivo não encontrado - {e.filename}")
	except Exception as e:
		print(f"Erro inesperado ao ler os arquivos txt: {e}")

	
	X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,28]].values
	#Y = df.iloc[:,[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]].values
	Y =  df.iloc[:,[24,25,26,27]].values

	escala_X = StandardScaler()
	escala_Y = StandardScaler()

	escala_X.fit(X)
	escala_Y.fit(Y)

	X_norm = escala_X.transform(X)
	Y_norm = escala_Y.transform(Y)

	return X_norm, Y_norm,escala_X,escala_Y

def criar_modelo():


	if os.path.exists(f'model_checkpoint.keras'):
		model = tf.keras.models.load_model('model_checkpoint.keras')
		print("O modelo foi carregado do checkpoint.")

	else:

		r2_metric = keras.metrics.R2Score()

		model = keras.Sequential([
		layers.Dense(13, activation='relu', input_dim=13),
		layers.Dense(65, activation='relu'),
		layers.Dropout(0.2),
		layers.Dense(65, activation='relu'),
		layers.Dropout(0.2),
		layers.Dense(65, activation='relu'),
		layers.Dropout(0.2),
		layers.Dense(65, activation='relu'),
		layers.Dropout(0.2),
		layers.Dense(65, activation='relu'),
		layers.Dropout(0.2),
		layers.Dense(4, activation='relu')
		])

		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
			loss='mse',
			metrics=['mae','mse', r2_metric])

	

	return model

def treinar():

	batch_size = 256
	print("Iniciando o programa")
	X_norm, Y_norm,escala_X,escala_Y = normalizar_dados()

	print("Iniciando a divisao de treino e teste")
	x_train, x_test, y_train, y_test = train_test_split(X_norm, Y_norm, test_size=0.25)
	print("Divisao realizada com sucesso")

	current_epoch_callback = CurrentEpochCallback()
	
	checkpoint_callback = keras.callbacks.ModelCheckpoint(
	filepath=f'model_checkpoint.keras',
	save_weights_only=False,
	save_best_only=True,
	monitor='val_loss',
	mode='min',
	verbose=1)

	print("Inicializando o modelo")
	model = criar_modelo()
	model.summary()

	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000,min_delta=1e-8,restore_best_weights=True)

	print("Inicializando o treinamento")
	history = model.fit(x_train,y_train, epochs=10000000, validation_data=(x_test,y_test),batch_size=batch_size,callbacks=[early_stop,current_epoch_callback,checkpoint_callback,tensorboard])
	print(model.evaluate(x_test,y_test))
	

	pd.DataFrame(history.history).plot(figsize=(8,5))
	plt.grid(True)
	plt.gca().set_ylim(0,1) # set the vertical range to [0-1]
	plt.show()

	#print(escala_Y.inverse_transform(model.predict(x_test)))

	#plot_history(history)

def plot_history(history):

	hist = pd.DataFrame(history.history)

	print(hist.head)

	hist['epoch'] = history.epoch

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [MPG]')
	plt.plot(hist['epoch'], hist['mae'],
					 label='Train Error')
	plt.plot(hist['epoch'], hist['val_mae'],
					 label = 'Val Error')
	plt.grid()
	plt.legend()

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Square Error [$MPG^2$]')
	plt.plot(hist['epoch'], hist['mse'],
					 label='Train Error')
	plt.plot(hist['epoch'], hist['val_mse'],
					 label = 'Val Error')
	plt.grid()
	plt.legend()
	plt.show()


treinar()
