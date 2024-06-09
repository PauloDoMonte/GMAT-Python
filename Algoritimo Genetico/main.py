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

def propulsao_1():
    # Nome do arquivo onde os resultados serão salvos
    arquivo = f'algoritimo_genetico_parte{int(sys.argv[1])}.txt'

    # Cria o arquivo se ele não existir e escreve o cabeçalho
    if not os.path.exists(arquivo):
        with open(arquivo, 'a') as arquivo_:
            arquivo_.write("dr,dv,vx,vy,vz,t1,t2\n")

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
        n_0 = (gmat.GetRuntimeObject("n_sma0").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_ecc0").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_inc0").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_raan0").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_aop0").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_ta0").GetNumber("Value"))

        n_f = (gmat.GetRuntimeObject("n_smaf").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_eccf").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_incf").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_raanf").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_aopf").GetNumber("Value"), 
               gmat.GetRuntimeObject("n_taf").GetNumber("Value"))

        d_0 = (gmat.GetRuntimeObject("d_sma0").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_ecc0").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_inc0").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_raan0").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_aop0").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_ta0").GetNumber("Value"))

        d_f = (gmat.GetRuntimeObject("d_smaf").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_eccf").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_incf").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_raanf").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_aopf").GetNumber("Value"), 
               gmat.GetRuntimeObject("d_taf").GetNumber("Value"))

        # Recupera as distâncias e velocidades relativas
        dr = gmat.GetRuntimeObject("dr").GetNumber("Value")
        dv = gmat.GetRuntimeObject("dv").GetNumber("Value")

        # Calcula os periápsides (rp) dos corpos
        rp_n = n_f[0] * (1 - n_f[1])
        rp_d = d_f[0] * (1 - d_f[1])

        # Verifica se as condições de rp são satisfatórias e, se sim, grava os resultados no arquivo
        if rp_n > 6317 and n_f[1] < 1 and rp_d > 6317 and d_f[1] < 1:
                # Ler a última linha do arquivo
                with open(arquivo, 'r') as arquivo_:
                    linhas = arquivo_.readlines()
                    ultima_linha = linhas[-1] if len(linhas) > 1 else None

                # Verificar se a última linha existe e comparar os valores de dr
                if ultima_linha:
                    ultima_dr = float(ultima_linha.split(',')[0])
                    if dr < ultima_dr:
                        print(f"Melhorou: {round((((ultima_dr/dr)*100)-100),3)}%")
                        with open(arquivo, 'a') as arquivo_:
                            arquivo_.write(f"{dr},{dv},{individuo[0]},{individuo[1]},{individuo[2]},{individuo[3]},{individuo[4]}\n")
                else:
                    # Se não há última linha, escreva a nova linha
                    with open(arquivo, 'a') as arquivo_:
                        arquivo_.write(f"{dr},{dv},{individuo[0]},{individuo[1]},{individuo[2]},{individuo[3]},{individuo[4]}\n")

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

    # Configurações do DEAP
    toolbox = base.Toolbox()
    
    # Criação dos tipos de fitness e indivíduo no DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("EstrIndividuo", list, fitness=creator.FitnessMin)

    # Intervalos para geração de indivíduos
    dv = [-2.0, 5.0]  # Range para a variação de velocidade causada pela propulsão
    dt = [1, 10000]  # Range do tempo que vai acontecer pré e pós propulsões

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

    print("")
    # Executa o algoritmo genético
    resultado, log = algorithms.eaSimple(pop, toolbox, cxpb=cx_pb_, mutpb=mutpb_, halloffame=hof, stats=estatistica,
                                         ngen=100000, verbose=True)

    # Tempo final
    depois = time.time()

    # Imprime o tempo total de execução
    print(f"Terminou {depois - antes}")

# Chama a função principal para iniciar a otimização
propulsao_1()
