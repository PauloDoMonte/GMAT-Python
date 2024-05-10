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


with open('propulsao1.txt', 'w') as arquivo:  # 'a' para append (adicionar)
	arquivo.write("dr,dv,vx1,vy1,vz1,t1,t2\n")

def individuos(intervalos):
	t01_x,t01_y,t01_z,t1,t2,= intervalos
	individuo = [
		np.random.uniform(t01_x[0],t01_x[1]),
		np.random.uniform(t01_y[0],t01_y[1]),
		np.random.uniform(t01_z[0],t01_z[1]),
		np.random.uniform(t1[0],t1[1]),
		np.random.uniform(t2[0],t2[1]),
	]
	return individuo

def aptidao(individuo):
	
	T01.SetField("Element1", individuo[0])
	T01.SetField("Element2", individuo[1])
	T01.SetField("Element3", individuo[2])

	t01_start.SetField("Value", individuo[3])
	t02_start.SetField("Value", individuo[4])

	gmat.RunScript()

	nx,ny,nz 	= gmat.GetRuntimeObject("nx").GetNumber("Value"),gmat.GetRuntimeObject("ny").GetNumber("Value"),gmat.GetRuntimeObject("nz").GetNumber("Value")
	dx,dy,dz 	= gmat.GetRuntimeObject("dx").GetNumber("Value"),gmat.GetRuntimeObject("dy").GetNumber("Value"),gmat.GetRuntimeObject("dz").GetNumber("Value")
	nvx,nvy,nvz = gmat.GetRuntimeObject("nvx").GetNumber("Value"),gmat.GetRuntimeObject("nvy").GetNumber("Value"),gmat.GetRuntimeObject("nvz").GetNumber("Value")
	dvx,dvy,dvz = gmat.GetRuntimeObject("dvx").GetNumber("Value"),gmat.GetRuntimeObject("dvy").GetNumber("Value"),gmat.GetRuntimeObject("dvz").GetNumber("Value")
	
	d_nave 	= np.sqrt((nx**2)+(ny**2)+(nz**2))
	d_det 	= np.sqrt((dx**2)+(dy**2)+(dz**2))
	v_nave 	= np.sqrt((nvx**2)+(nvy**2)+(nvz**2))
	v_det 	= np.sqrt((dvx**2)+(dvy**2)+(dvz**2))

	v1 = gmat.GetRuntimeObject("v1").GetNumber("Value")
	v2 = gmat.GetRuntimeObject("v2").GetNumber("Value")

	if(v1 < 6317 or v2 < 6317): d_nave = 1e18;v_nave = 1e18

	dr 	= np.sqrt((nx-dx)**2+(ny-dy)**2+(nz-dz)**2)
	dv 	= np.sqrt((nvx-dvx)**2+(nvy-dvy)**2+(nvz-dvz)**2)

	if(dr < 1): 
		with open('propulsao1.txt', 'a') as arquivo:  # 'a' para append (adicionar)
			arquivo.write(f"{dr},{dv},{individuo[0]},{individuo[1]},{individuo[2]},{individuo[3]},{individuo[4]}\n")
	return dr,

Script_missao = "propulsao1"
gmat.LoadScript(Script_missao + ".script")

T01 = gmat.GetObject("T01") # Variaveis da velocidade primeira queima (vx,vy,vz) 

t01_start = gmat.GetObject("T01_start") # Tempo que inicia a primeira queima
t02_start = gmat.GetObject("T02_start") # Tempo que inicia a segunda queima

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("EstrIndividuo", list, fitness=creator.FitnessMin)

dv = [-2.0,5.] # Range para a variação de velocidade causada pela propulsão
dt = [1,10000] # Range do tempo que vai acontecer pré e pós propulsões

toolbox.register("Genes", individuos, intervalos=[dv,dv,dv,dt,dt])
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
estatistica = tools.MultiStatistics(dr=dr)

estatistica.register('Média',np.mean)
estatistica.register('Min',np.min)
estatistica.register('Max',np.max)

hof = tools.HallOfFame(1)

antes = time.time()

print("")
resultado, log = algorithms.eaSimple(pop, toolbox,cxpb=cx_pb_,mutpb=mutpb_,halloffame=hof,stats=estatistica,ngen=100000,verbose=True)

depois = time.time()

print(f"Terminou {depois-antes}")
