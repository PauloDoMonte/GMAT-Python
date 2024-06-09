import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('algoritimo_genetico_parte100.txt')

plt.scatter(df.index,df['dr'])
plt.ylabel("Distância Final Relativa [Km]")
plt.xlabel("Quantidade de rodadas")
plt.grid()


plt.show()