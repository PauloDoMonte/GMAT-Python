'''
	Esse programa carrega a bilioteca que manipula o GMAT da Nasa (load_gmat). Carrega um script gerado após eu montar uma simulação
	no GMAT, esse script tem a configuração da Terra, uma nave e um detrito orbitando a Terra, com a nave em uma orbita menor.
	O objetivo do programa é calcular duas propulsões possiveis T01 e T02 com suas compoventes(vx,vy,vz)(1,2) e os tempos t1,t2,t3
	onde t1 é o tempo antes de T01, t2 é o tempo pós T01 e antes de T02 e t3 é o tempo pós T02. Através de um algoritimo genetico, 
	buscamos bons parametros para propulsão e tempos para ocorrer um rendezvous (dr e dv = 0) onde a distancia e velocidade relativa
	entre nave e detrito seja bem proxima a zero.
'''

from load_gmat import *
from math import *
import numpy as np
from deap import creator, base, tools, algorithms
import random, time, os, sys

with open('melhores.txt', 'w') as arquivo:  # 'a' para append (adicionar)
	arquivo.write("dr,dv,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3,vx4,vy4,vz4,t1,t2,t3,t4,t5\n")

def individuos(intervalos):
	t01_x,t01_y,t01_z,t02_x,t02_y,t02_z, t03_x,t03_y,t03_z, t04_x,t04_y,t04_z,t1,t2, t3, t4,t5 = intervalos
	individuo = [
		np.random.uniform(t01_x[0],t01_x[1]),
		np.random.uniform(t01_y[0],t01_y[1]),
		np.random.uniform(t01_z[0],t01_z[1]),
		np.random.uniform(t02_x[0],t02_x[1]),
		np.random.uniform(t02_y[0],t02_y[1]),
		np.random.uniform(t02_z[0],t02_z[1]),
		np.random.uniform(t03_x[0],t03_x[1]),
		np.random.uniform(t03_y[0],t03_y[1]),
		np.random.uniform(t03_z[0],t03_z[1]),
		np.random.uniform(t04_x[0],t04_x[1]),
		np.random.uniform(t04_y[0],t04_y[1]),
		np.random.uniform(t04_z[0],t04_z[1]),
		np.random.uniform(t1[0],t1[1]),
		np.random.uniform(t2[0],t2[1]),
		np.random.uniform(t3[0],t3[1]),
		np.random.uniform(t4[0],t4[1]),
		np.random.uniform(t5[0],t5[1])
	]
	return individuo

def aptidao(individuo):
	
	T01.SetField("Element1", individuo[0])
	T01.SetField("Element2", individuo[1])
	T01.SetField("Element3", individuo[2])

	T02.SetField("Element1", individuo[3])
	T02.SetField("Element2", individuo[4])
	T02.SetField("Element3", individuo[5])

	T03.SetField("Element1", individuo[6])
	T03.SetField("Element2", individuo[7])
	T03.SetField("Element3", individuo[8])

	T04.SetField("Element1", individuo[9])
	T04.SetField("Element2", individuo[10])
	T04.SetField("Element3", individuo[11])

	t01_start.SetField("Value", individuo[12])
	t02_start.SetField("Value", individuo[13])
	t03_start.SetField("Value", individuo[14])
	t04_start.SetField("Value", individuo[15])
	t05_start.SetField("Value", individuo[16])

	gmat.RunScript()

	nx,ny,nz = gmat.GetRuntimeObject("nx").GetNumber("Value"),gmat.GetRuntimeObject("ny").GetNumber("Value"),gmat.GetRuntimeObject("nz").GetNumber("Value")
	dx,dy,dz = gmat.GetRuntimeObject("dx").GetNumber("Value"),gmat.GetRuntimeObject("dy").GetNumber("Value"),gmat.GetRuntimeObject("dz").GetNumber("Value")
	
	nvx,nvy,nvz = gmat.GetRuntimeObject("nvx").GetNumber("Value"),gmat.GetRuntimeObject("nvy").GetNumber("Value"),gmat.GetRuntimeObject("nvz").GetNumber("Value")
	dvx,dvy,dvz = gmat.GetRuntimeObject("dvx").GetNumber("Value"),gmat.GetRuntimeObject("dvy").GetNumber("Value"),gmat.GetRuntimeObject("dvz").GetNumber("Value")
	
	d_nave = np.sqrt((nx**2)+(ny**2)+(nz**2))
	d_det = np.sqrt((dx**2)+(dy**2)+(dz**2))

	v_nave = np.sqrt((nvx**2)+(nvy**2)+(nvz**2))
	v_det = np.sqrt((dvx**2)+(dvy**2)+(dvz**2))

	v1 = gmat.GetRuntimeObject("v1").GetNumber("Value")
	v2 = gmat.GetRuntimeObject("v2").GetNumber("Value")
	v3 = gmat.GetRuntimeObject("v3").GetNumber("Value")
	v4 = gmat.GetRuntimeObject("v4").GetNumber("Value")

	if(v1 < 6317 or v2 < 6317 or v3 < 6317 or v4 < 6317 or d_nave < 6317):
		d_nave = 1e18; v_nave = 1e18

	dr = abs(d_nave-d_det)
	dv = abs(v_nave-v_det)

	if(dr < 1 and dv < 1): 
		with open('melhores.txt', 'a') as arquivo:  # 'a' para append (adicionar)
			arquivo.write(f"{dr},{dv},{individuo[0]},{individuo[1]},{individuo[2]},{individuo[3]},{individuo[4]},{individuo[5]},{individuo[6]},{individuo[7]},{individuo[8]},{individuo[9]},{individuo[10]},{individuo[11]},{individuo[12]},{individuo[13]},{individuo[14]},{individuo[15]},{individuo[16]}\n")
	return dr,dv,

Script_missao = "uefs"
gmat.LoadScript(Script_missao + ".script")

T01 = gmat.GetObject("T01") # Variaveis da velocidade primeira queima (vx,vy,vz) 
T02 = gmat.GetObject("T02") # Variaveis da velocidade segunda queima (vx,vy,vz)
T03 = gmat.GetObject("T03") # Variaveis da velocidade terceira queima (vx,vy,vz)
T04 = gmat.GetObject("T04") # Variaveis da velocidade quarta queima (vx,vy,vz)

t01_start = gmat.GetObject("T01_start") # Tempo que inicia a primeira queima
t02_start = gmat.GetObject("T02_start") # Tempo que inicia a segunda queima
t03_start = gmat.GetObject("T03_start") # Tempo que inicia a segunda queima
t04_start = gmat.GetObject("T04_start") # Tempo que inicia a segunda queima
t05_start = gmat.GetObject("T05_start") # Tempo que inicia a segunda queima


toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("EstrIndividuo", list, fitness=creator.FitnessMin)

dv = [-2.0,2.] # Range para a variação de velocidade causada pela propulsão
dt = [1,100000] # Range do tempo que vai acontecer pré e pós propulsões

toolbox.register("Genes", individuos, intervalos=[dv,dv,dv,dv,dv,dv,dv,dv,dv,dv,dv,dv, dt,dt,dt,dt,dt])
toolbox.register("Individuos", tools.initIterate, creator.EstrIndividuo, toolbox.Genes)
toolbox.register("Populacao", tools.initRepeat, list, toolbox.Individuos)

pop = toolbox.Populacao(n=int(sys.argv[1]))

cx_pb_, mutpb_ = 0.8,0.2

toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=mutpb_)
toolbox.register("select", tools.selTournament, tournsize=2)
#toolbox.register("select",tools.selNSGA2)
toolbox.register("evaluate", aptidao)

dr = tools.Statistics(key=lambda individuo: individuo.fitness.values[0])
dv = tools.Statistics(key=lambda individuo: individuo.fitness.values[1])
estatistica = tools.MultiStatistics(dr=dr, dv=dv)

estatistica.register('Média',np.mean)
estatistica.register('Min',np.min)
estatistica.register('Max',np.max)

hof = tools.HallOfFame(1)

antes = time.time()

print("")
resultado, log = algorithms.eaSimple(pop, toolbox,cxpb=cx_pb_,mutpb=mutpb_,halloffame=hof,stats=estatistica,ngen=1000000,verbose=True)

depois = time.time()
print("")
print("="*50)
print(f"""
Após {round(((depois-antes)/60),2)} minutos de simulacao, temos:
t1:\t{hof[0][6]}
TOI:\t({hof[0][0]},{hof[0][1]},{hof[0][2]}) (km/s)
t2:\t{hof[0][7]}
TOII:\t({hof[0][3]},{hof[0][4]},{hof[0][5]}) (km/s)
t3:\t{hof[0][8]}

dr, dv: {aptidao(hof[0])}

Rendezvous realizado com sucesso :)

""")
print("="*50)
print("")
