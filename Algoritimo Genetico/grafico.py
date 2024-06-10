import pandas as pd
import matplotlib.pyplot as plt

def grafico_1():

	# Carregar os dados do arquivo CSV
	df1 = pd.read_csv('algoritimo_genetico_parte1000.txt')
	df2 = pd.read_csv('algoritimo_genetico_parte10000.txt')

	# Criar a figura e os eixos para os dois plots
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

	# Primeiro plot: scatter plot de 'dr' vs. index
	ax1.scatter(df1.index, df1['dr'])
	ax1.set_ylabel("Distância Final Relativa [Km]")
	ax1.set_xlabel("Quantidade de rodadas")
	ax1.grid()

	# Segundo plot: scatter plot de 'dv' vs. index
	ax2.scatter(df2.index, df2['dr'])
	ax2.set_ylabel("Distância Final Relativa [Km]")
	ax2.set_xlabel("Quantidade de rodadas")
	ax2.grid()

	# Mostrar os plots
	plt.tight_layout()
	plt.show()


def grafico_2():
	# Carregar os dados dos arquivos CSV
	df1 = pd.read_csv('algoritimo_genetico_parte1000.txt')
	df2 = pd.read_csv('algoritimo_genetico_parte10000.txt')

	# Criar a figura e os eixos para os quatro plots
	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 12))

	# Primeiro plot: scatter plot de 'dr' vs. index para df1
	ax1.scatter(df1.index, df1['vx'])
	ax1.set_ylabel("Vx")
	ax1.set_xlabel("Quantidade de rodadas")
	ax1.set_title("Dataset 1 - Distância Final Relativa")
	ax1.grid()

	# Segundo plot: scatter plot de 'dv' vs. index para df1
	ax2.scatter(df1.index, df1['vy'])
	ax2.set_ylabel("Vy")
	ax2.set_xlabel("Quantidade de rodadas")
	ax2.set_title("Dataset 1 - Velocidade Final Relativa")
	ax2.grid()

	# Segundo plot: scatter plot de 'dv' vs. index para df1
	ax3.scatter(df1.index, df1['vz'])
	ax3.set_ylabel("Vz")
	ax3.set_xlabel("Quantidade de rodadas")
	ax3.set_title("Dataset 1 - Velocidade Final Relativa")
	ax3.grid()

	# Terceiro plot: scatter plot de 'dr' vs. index para df2
	ax4.scatter(df2.index, df2['vx'])
	ax4.set_ylabel("Vx")
	ax4.set_xlabel("Quantidade de rodadas")
	ax4.set_title("Dataset 2 - Distância Final Relativa")
	ax4.grid()

	# Quarto plot: scatter plot de 'dv' vs. index para df2
	ax5.scatter(df2.index, df2['vy'])
	ax5.set_ylabel("Vy")
	ax5.set_xlabel("Quantidade de rodadas")
	ax5.set_title("Dataset 2 - Velocidade Final Relativa")
	ax5.grid()

	# Quarto plot: scatter plot de 'dv' vs. index para df2
	ax6.scatter(df2.index, df2['vz'])
	ax6.set_ylabel("Vz")
	ax6.set_xlabel("Quantidade de rodadas")
	ax6.set_title("Dataset 2 - Velocidade Final Relativa")
	ax6.grid()

	# Ajustar automaticamente os espaçamentos entre os plots
	plt.tight_layout()

	# Mostrar os plots
	plt.show()

def grafico_3():
	# Carregar os dados dos arquivos CSV
	df1_ = pd.read_csv('algoritimo_genetico_parte1000.txt')
	df2_ = pd.read_csv('algoritimo_genetico_parte10000.txt')

	df1 = df1_.query('dr < 1000')
	df2 = df2_.query('dr < 1000')

	# Criar a figura e os eixos para os quatro plots
	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 12))

	# Primeiro plot: scatter plot de 'dr' vs. index para df1
	ax1.scatter(df1['vx'],df1['dr'])
	ax1.set_ylabel("Distância")
	ax1.set_xlabel("Vx")
	ax1.set_title("Dataset 1 - Distância Final Relativa")
	ax1.grid()

	# Segundo plot: scatter plot de 'dv' vs. index para df1
	ax2.scatter(df1['vy'],df1['dr'])
	ax2.set_ylabel("Distância")
	ax2.set_xlabel("Vy")
	ax2.set_title("Dataset 1 - Velocidade Final Relativa")
	ax2.grid()

	# Segundo plot: scatter plot de 'dv' vs. index para df1
	ax3.scatter(df1['vz'],df1['dr'])
	ax3.set_ylabel("Distância")
	ax3.set_xlabel("Vz")
	ax3.set_title("Dataset 1 - Velocidade Final Relativa")
	ax3.grid()

	# Terceiro plot: scatter plot de 'dr' vs. index para df2
	ax4.scatter(df2['vx'],df2['dr'])
	ax4.set_ylabel("Distância")
	ax4.set_xlabel("Vx")
	ax4.set_title("Dataset 2 - Distância Final Relativa")
	ax4.grid()

	# Quarto plot: scatter plot de 'dv' vs. index para df2
	ax5.scatter(df2['vy'],df2['dr'])
	ax5.set_ylabel("Distância")
	ax5.set_xlabel("Vy")
	ax5.set_title("Dataset 2 - Velocidade Final Relativa")
	ax5.grid()

	# Quarto plot: scatter plot de 'dv' vs. index para df2
	ax6.scatter(df2['vz'],df2['dr'])
	ax6.set_ylabel("Distância")
	ax6.set_xlabel("Vz")
	ax6.set_title("Dataset 2 - Velocidade Final Relativa")
	ax6.grid()

	# Ajustar automaticamente os espaçamentos entre os plots
	plt.tight_layout()

	# Mostrar os plots
	plt.show()

grafico_1()
grafico_2()
grafico_3()