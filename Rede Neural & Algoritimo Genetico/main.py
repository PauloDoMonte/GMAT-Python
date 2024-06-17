import random, time, os, sys

# Adiciona o diretório da API ao caminho do sistema
Api = "/home/domonte/R2022a/api"
sys.path.insert(1, Api)

import numpy as np
from load_gmat import *
from deap import creator, base, tools, algorithms

"""
Programa de Algoritmo Genético

Consiste em utilizar algoritmo genético junto do GMAT da NASA, com o objetivo
de buscar bons parâmetros de propulsão (vx, vy, vz) e em certos tempos (t1, t2) 
para conseguir primeiramente que o corpo 1 alcance o corpo 2 em órbita. Com a futura função propulsao_2,
pretende-se circularizar a órbita e terminar o rendezvous.
Na propulsao_1, buscamos diminuir o parâmetro dr (distância final relativa) e, na futura função propulsao_2,
vamos buscar diminuir dr e dv (velocidade final relativa) a fim de conseguir um rendezvous.
"""

arquivo = f'algoritimo_genetico_parte_1.txt'

def propulsao_1():

    verifica_pop = []
	
    # Cria o arquivo se ele não existir e escreve o cabeçalho
    if not os.path.exists(arquivo):
        with open(arquivo, 'a') as arquivo_:
            arquivo_.write("nx0,ny0,nz0,dx0,dy0,dz0,nvx0,nvy0,nz0,dvx0,dy0,dvz0,n_dm,d_dm,vx,vy,vz,t1,t2,dr\n")

    # Função para gerar indivíduos aleatórios dentro dos intervalos fornecidos
    def individuos(intervalos):
        t01_x, t01_y, t01_z, t1, t2 = intervalos
        individuo = [
            np.random.uniform(t01_x[0], t01_x[1]),
            np.random.uniform(t01_y[0], t01_y[1]),
            np.random.uniform(t01_z[0], t01_z[1]),
            np.random.uniform(t1[0], t1[1]),
            np.random.uniform(t2[0], t2[1]),
        ]
        return individuo

    # Função para avaliar a aptidão de um indivíduo
    def aptidao(individuo):
        # Define os parâmetros de propulsão e tempos no GMAT
        T01.SetField("Element1", individuo[0])
        T01.SetField("Element2", individuo[1])
        T01.SetField("Element3", individuo[2])

        t01_start.SetField("Value", individuo[3])
        t02_start.SetField("Value", individuo[4])

        # Executa o script do GMAT
        gmat.RunScript()

        # Recupera os elementos orbitais iniciais e finais dos corpos
        nr0 = [gmat.GetRuntimeObject("nx0").GetNumber("Value"),gmat.GetRuntimeObject("ny0").GetNumber("Value"),gmat.GetRuntimeObject("nz0").GetNumber("Value")]
        dr0 = [gmat.GetRuntimeObject("dx0").GetNumber("Value"),gmat.GetRuntimeObject("dy0").GetNumber("Value"),gmat.GetRuntimeObject("dz0").GetNumber("Value")]
        
        nv0 = [gmat.GetRuntimeObject("nvx0").GetNumber("Value"),gmat.GetRuntimeObject("nvy0").GetNumber("Value"),gmat.GetRuntimeObject("nvz0").GetNumber("Value")]
        dv0 = [gmat.GetRuntimeObject("dvx0").GetNumber("Value"),gmat.GetRuntimeObject("dvy0").GetNumber("Value"),gmat.GetRuntimeObject("dvz0").GetNumber("Value")]
        
        
        n_a = gmat.GetRuntimeObject("n_a").GetNumber("Value")
        n_e = gmat.GetRuntimeObject("n_e").GetNumber("Value")
        d_a = gmat.GetRuntimeObject("d_a").GetNumber("Value")
        d_e = gmat.GetRuntimeObject("d_e").GetNumber("Value")
        
        
        # Recupera as distâncias e velocidades relativas
        dr = gmat.GetRuntimeObject("dr").GetNumber("Value")

        # Calcula os periápsides (rp) dos corpos
        rp_n = n_a * (1 - n_e)
        rp_d = d_a * (1 - d_e)
        
        # Verifica se as condições de rp são satisfatórias e, se sim, grava os resultados no arquivo
        if rp_n > 6317 and n_e < 1 and rp_d > 6317 and d_e < 1:
            if(dr < 100):
                with open(arquivo, 'a') as arquivo_:
                    arquivo_.write(f"{nr0[0]},{nr0[1]},{nr0[2]},{dr0[0]},{dr0[1]},{dr0[2]},{nv0[0]},{nv0[1]},{nv0[2]},{dv0[0]},{dv0[1]},{dv0[2]},{dry_massa_n},{dry_massa_d},{individuo[0]},{individuo[1]},{individuo[2]},{individuo[3]},{individuo[4]},{dr}\n")
                
        else:
            dr = 1e18

        return dr,

    # Carrega o script da missão do GMAT
    Script_missao = "rede_neural"
    gmat.LoadScript(Script_missao + ".script")

    # Define os objetos de propulsão e tempos no GMAT
    T01 = gmat.GetObject("T01")  # Variáveis de velocidade da primeira queima (vx, vy, vz)
    t01_start = gmat.GetObject("T01_start")  # Tempo que inicia a primeira queima
    t02_start = gmat.GetObject("T02_start")  # Tempo que inicia a segunda queima
    
    Nave = gmat.GetObject("Nave")
    Det = gmat.GetObject("Det")
    
    a_nave = 300
    a_det = np.random.uniform(1000,30000)
    
    e_nave = np.random.uniform(0.001,0.9)
    e_det = np.random.uniform(0.001,0.9)
    
    try:
    
        Nave.SetField("SMA",(6371+a_nave))
        Nave.SetField("ECC",e_nave)
    
    except:
        try:
            Nave.SetField("SMA",(6371+a_nave))
        except:
            try:
                Nave.SetField("ECC",e_nave)
            except: pass
            
    try:
    
        Det.SetField("SMA",(6371+a_det))
        Det.SetField("ECC",e_det)
    
    except:
        try:
            Det.SetField("SMA",(6371+a_det))
        except:
            try:
                Det.SetField("ECC",e_det)
            except: pass
        
    nave_inc 	= np.random.uniform(0,180)
    nave_raan 	= np.random.uniform(0,360)
    nave_aop 	= np.random.uniform(0,360)
    nave_f 		= np.random.uniform(0,360)

	
    det_inc 	= np.random.uniform(0,180)
    det_raan 	= np.random.uniform(0,360)
    det_aop 	= np.random.uniform(0,360)
    det_f 		= np.random.uniform(0,360)
    
    dry_massa_n = np.random.uniform(100,10000)
    dry_massa_d = np.random.uniform(1,10000)
    
    Nave.SetField("INC",nave_inc)
    Nave.SetField("RAAN",nave_raan)
    Nave.SetField("AOP",nave_aop)
    Nave.SetField("TA",nave_f)
    Nave.SetField("DryMass",dry_massa_n)

    Det.SetField("INC",det_inc)
    Det.SetField("RAAN",det_raan)
    Det.SetField("AOP",det_aop)
    Det.SetField("TA",det_f)
    Det.SetField("DryMass",dry_massa_d)

    # Configurações do DEAP
    toolbox = base.Toolbox()
    
    # Criação dos tipos de fitness e indivíduo no DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("EstrIndividuo", list, fitness=creator.FitnessMin)

    # Intervalos para geração de indivíduos
    dv = [-10./np.sqrt(3), 10./np.sqrt(3)]  # Range para a variação de velocidade causada pela propulsão
    dt = [1, 86400/2]  # Range do tempo que vai acontecer pré e pós propulsões

    # Registra os operadores genéticos no toolbox
    toolbox.register("Genes", individuos, intervalos=[dv, dv, dv, dt, dt])
    toolbox.register("Individuos", tools.initIterate, creator.EstrIndividuo, toolbox.Genes)
    toolbox.register("Populacao", tools.initRepeat, list, toolbox.Individuos)

    # Cria a população inicial
    pop = toolbox.Populacao(n=int(sys.argv[1]))

    # Probabilidades de cruzamento e mutação
    cx_pb_, mutpb_ = 0.8, 0.2

    # Define os operadores de cruzamento, mutação e seleção
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=mutpb_)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", aptidao)

    # Estatísticas a serem coletadas durante a execução do algoritmo genético
    dr_stats = tools.Statistics(key=lambda individuo: individuo.fitness.values[0])
    estatistica = tools.MultiStatistics(dr=dr_stats)

    estatistica.register('Média', np.mean)
    estatistica.register('Min', np.min)
    estatistica.register('Max', np.max)

    # Hall of Fame para armazenar os melhores indivíduos
    hof = tools.HallOfFame(1)

    # Tempo inicial
    antes = time.time()

    print(f"Semi-Eixo\nNave:{a_nave}\tDet:{a_det}")
    # Executa o algoritmo genético
    resultado, log = algorithms.eaSimple(pop, toolbox, cxpb=cx_pb_, mutpb=mutpb_, halloffame=hof, stats=estatistica,
                                         ngen=1000, verbose=True)

    # Tempo final
    depois = time.time()

    # Imprime o tempo total de execução
    print(f"Terminou {depois - antes}")

# Chama a função principal para iniciar a otimização
while(True):
	propulsao_1()
