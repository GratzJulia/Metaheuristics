import time
import numpy as np
from random import randint, sample, uniform
import matplotlib.pyplot as plt
import networkx as nx


# Função para ler o arquivo no formato DIMACS
def read_DIMACS(filepath):
    """
    Lê um arquivo DIMACS e retorna:
    - Número de vértices (v)
    - Número de arestas (a)
    - Lista de arestas [(origem, destino)]
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    v, a = 0, 0
    arestas = []

    for line in lines:
        if line.startswith('p'):  # Linha que contém 'p edge num_vertices num_arestas'
            _, _, v, a = line.split()
            v, a = int(v), int(a)
        elif line.startswith('e'):  # Linha que contém 'e origem destino'
            _, origem, destino = line.split()
            arestas.append((int(origem), int(destino)))

    return v, a, arestas


# Classe que representa o grafo
class Grafo:
    def __init__(self, v):
        self.V = v
        self.arestas = []

    def add_aresta(self, origem, destino, peso=0.0):
        self.arestas.append(Aresta(origem, destino, peso))


class Aresta:
    def __init__(self, origem, destino, peso):
        self.origem = origem
        self.destino = destino
        self.peso = peso


# Função para plotar o grafo
def printGrafo(v: int, a: int, arestas: list, cromossomo: list = None):
    G = nx.Graph()
    [G.add_node(i + 1) for i in range(v)]
    [G.add_edge(aresta[0], aresta[1]) for aresta in arestas]

    if cromossomo:
        color_map = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta']
        colors = [color_map[cor - 1] if cor <= len(color_map) else 'white' for cor in cromossomo]
    else:
        colors = ['white'] * v

    nx.draw(G, with_labels=True, node_color=colors, node_size=400, edgecolors='black', linewidths=1)
    plt.show()


# Classe para o Algoritmo Genético
class AlgoritmoGenetico:
    def __init__(self, grafo: Grafo):
        self.grafo = grafo
        self.tamanho_populacao = 2 * grafo.V
        self.len_elites = max(1, int(0.2 * self.tamanho_populacao))  # At least one elite

    def FO(self, cromossomo):
        penalidade_aresta = 0
        cor_count = {}

        for aresta in self.grafo.arestas:
            if cromossomo[aresta.origem - 1] == cromossomo[aresta.destino - 1]:
                penalidade_aresta += 20

        for cor in cromossomo:
            cor_count[cor] = cor_count.get(cor, 0) + 1

        max_count = max(cor_count.values(), default=0)
        min_count = min(cor_count.values(), default=0)
        desequilibrio = max_count - min_count

        return penalidade_aresta + 0.1 * len(cor_count) + 0.5 * desequilibrio

    def fitness(self, cromossomo):
        return -1 * self.FO(cromossomo)

    def construtivo_aleatorio(self):
        return [[randint(1, self.grafo.V) for _ in range(self.grafo.V)] for _ in range(self.tamanho_populacao)]

    def set_elite(self, populacao: list):
        populacao.sort(key=lambda c: self.FO(c))
        return populacao[:self.len_elites]

    def crossover(self, pai1, pai2):
        ponto_corte = randint(1, len(pai1) - 1)
        return pai1[:ponto_corte] + pai2[ponto_corte:]

    def mutacao(self, cromossomo):
        idx = randint(0, len(cromossomo) - 1)
        cromossomo[idx] = randint(1, self.grafo.V)

    def selecionar_pais_torneio(self, populacao, fitness_populacao):
        pais = []
        for _ in range(len(populacao) // 2):
            selecionados = sample(list(zip(populacao, fitness_populacao)), 2)
            pais.append(max(selecionados, key=lambda x: x[1])[0])

        if len(pais) % 2 != 0:
            pais.pop()
        return pais

    def selecionar_pais_roleta(self, populacao, fitness_populacao):
        soma_fitness = sum(fitness_populacao)
        if soma_fitness == 0:
            probabilidades = [1 / len(fitness_populacao) for _ in fitness_populacao]
        else:
            probabilidades = [fitness / soma_fitness for fitness in fitness_populacao]

        pais = []
        for _ in range(len(populacao) // 2):
            r = uniform(0, 1)
            acumulador = 0
            for i, prob in enumerate(probabilidades):
                acumulador += prob
                if r <= acumulador:
                    pais.append(populacao[i])
                    break
        return pais

    def execute(self, geracoes=100, ordem_selecao=("torneio", "roleta")):
        populacao = self.construtivo_aleatorio()
        for geracao in range(geracoes):
            fitness_populacao = [self.fitness(cromossomo) for cromossomo in populacao]
            elites = self.set_elite(populacao)

            if ordem_selecao[0] == "torneio":
                melhores_pais = self.selecionar_pais_torneio(populacao, fitness_populacao)
                fitness_pais = [self.fitness(cromossomo) for cromossomo in melhores_pais]
                melhores_pais = self.selecionar_pais_roleta(melhores_pais, fitness_pais)
            else:
                melhores_pais = self.selecionar_pais_roleta(populacao, fitness_populacao)
                fitness_pais = [self.fitness(cromossomo) for cromossomo in melhores_pais]
                melhores_pais = self.selecionar_pais_torneio(melhores_pais, fitness_pais)

            if len(melhores_pais) % 2 != 0:
                melhores_pais.pop()

            nova_populacao = elites.copy()
            for i in range(0, len(melhores_pais), 2):
                pai1, pai2 = melhores_pais[i], melhores_pais[i + 1]
                filho1 = self.crossover(pai1, pai2)
                filho2 = self.crossover(pai2, pai1)
                self.mutacao(filho1)
                self.mutacao(filho2)
                nova_populacao.extend([filho1, filho2])

            populacao = nova_populacao[:self.tamanho_populacao]

        melhor_individuo = min(populacao, key=lambda cromossomo: self.FO(cromossomo))
        return melhor_individuo


# Busca Tabu
def busca_tabu(cromossomo_inicial, ag, iteracoes=100, tamanho_tabu=10):
    melhor_solucao = cromossomo_inicial[:]
    melhor_fo = ag.FO(cromossomo_inicial)
    tabu_list = []
    solucao_atual = cromossomo_inicial[:]

    for _ in range(iteracoes):
        vizinhos = []
        for i in range(len(solucao_atual)):
            vizinho = solucao_atual[:]
            vizinho[i] = randint(1, ag.grafo.V)
            if vizinho not in tabu_list:
                vizinhos.append(vizinho)

        if not vizinhos:
            break

        melhor_vizinho = max(vizinhos, key=lambda c: -ag.FO(c))
        fo_vizinho = ag.FO(melhor_vizinho)

        if fo_vizinho > melhor_fo:
            melhor_fo = fo_vizinho
            melhor_solucao = melhor_vizinho[:]

        solucao_atual = melhor_vizinho[:]
        tabu_list.append(melhor_vizinho)
        if len(tabu_list) > tamanho_tabu:
            tabu_list.pop(0)

    return melhor_solucao


# Comparação dos Métodos
def comparar_metodos(ag, geracoes=10, iteracoes_tabu=100, ordens=[("torneio", "roleta"), ("roleta", "torneio")]):
    resultados = []

    for ordem in ordens:
        # Sem Busca Tabu
        tempos = []
        fitness_sem_tabu = []

        for _ in range(10):
            start_time = time.time()
            melhor_solucao = ag.execute(geracoes=geracoes, ordem_selecao=ordem)
            elapsed_time = time.time() - start_time

            tempos.append(elapsed_time)
            fitness_sem_tabu.append(-ag.FO(melhor_solucao))

        media_tempo = np.mean(tempos)
        desvio_tempo = np.std(tempos)
        media_fitness = np.mean(fitness_sem_tabu)
        desvio_fitness = np.std(fitness_sem_tabu)

        resultados.append({
            "metodo": f"{' -> '.join(ordem)} (Sem Busca Tabu)",
            "melhor_solucao": min(fitness_sem_tabu),
            "pior_solucao": max(fitness_sem_tabu),
            "media_fitness": media_fitness,
            "desvio_fitness": desvio_fitness,
            "media_tempo": media_tempo,
            "desvio_tempo": desvio_tempo,
        })

        # Com Busca Tabu
        tempos = []
        fitness_com_tabu = []

        for _ in range(10):
            start_time = time.time()
            melhor_solucao = ag.execute(geracoes=geracoes, ordem_selecao=ordem)
            melhor_solucao_tabu = busca_tabu(melhor_solucao, ag, iteracoes=iteracoes_tabu)
            elapsed_time = time.time() - start_time

            tempos.append(elapsed_time)
            fitness_com_tabu.append(-ag.FO(melhor_solucao_tabu))

        media_tempo = np.mean(tempos)
        desvio_tempo = np.std(tempos)
        media_fitness = np.mean(fitness_com_tabu)
        desvio_fitness = np.std(fitness_com_tabu)

        resultados.append({
            "metodo": f"{' -> '.join(ordem)} (Com Busca Tabu)",
            "melhor_solucao": min(fitness_com_tabu),
            "pior_solucao": max(fitness_com_tabu),
            "media_fitness": media_fitness,
            "desvio_fitness": desvio_fitness,
            "media_tempo": media_tempo,
            "desvio_tempo": desvio_tempo,
        })

    return resultados


# Apresentar Resultados
def apresentar_resultados(resultados):
    print("\nResultados Finais por Método:")
    for resultado in resultados:
        print(f"Método: {resultado['metodo']}")
        print(f"  Melhor Solução: {resultado['melhor_solucao']}")
        print(f"  Pior Solução: {resultado['pior_solucao']}")
        print(f"  Média das Soluções: {resultado['media_fitness']:.2f}")
        print(f"  Desvio Padrão das Soluções: {resultado['desvio_fitness']:.2f}")
        print(f"  Média dos Tempos: {resultado['media_tempo']:.4f} s")
        print(f"  Desvio Padrão dos Tempos: {resultado['desvio_tempo']:.4f} s")
        print("-" * 40)


# Main
if __name__ == "__main__":
    v, a, arestas = read_DIMACS('./input-data/S10.txt')
    g = Grafo(v)
    [g.add_aresta(aresta[0], aresta[1], 0.0) for aresta in arestas]

    ag = AlgoritmoGenetico(g)

    resultados = comparar_metodos(
        ag,
        geracoes=10,
        iteracoes_tabu=100,
        ordens=[("torneio", "roleta"), ("roleta", "torneio")]
    )

    apresentar_resultados(resultados)




'''import time
from random import randint, sample, uniform
import matplotlib.pyplot as plt
import networkx as nx
from representacao import Grafo
from ReadData import read_DIMACS

def printGrafo(v: int, a: int, arestas: list, cromossomo: list = None):
    G = nx.Graph()
    [G.add_node(i + 1) for i in range(v)]
    [G.add_edge(aresta[0], aresta[1]) for aresta in arestas]

    if cromossomo:
        color_map = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta']
        colors = [color_map[cor - 1] if cor <= len(color_map) else 'white' for cor in cromossomo]
    else:
        colors = ['white'] * v

    nx.draw(G, with_labels=True, node_color=colors, node_size=400, edgecolors='black', linewidths=1)
    plt.show()

class AlgoritmoGenetico:
    def __init__(self, grafo: Grafo):
        self.grafo = grafo
        self.tamanho_populacao = 2 * grafo.V
        self.len_elites = max(1, int(0.2 * self.tamanho_populacao))  # At least one elite

    def FO(self, cromossomo):
        penalidade_aresta = 0
        cor_count = {}

        for aresta in self.grafo.arestas:
            if cromossomo[aresta.origem - 1] == cromossomo[aresta.destino - 1]:
                penalidade_aresta += 20

        for cor in cromossomo:
            cor_count[cor] = cor_count.get(cor, 0) + 1

        max_count = max(cor_count.values(), default=0)
        min_count = min(cor_count.values(), default=0)
        desequilibrio = max_count - min_count

        return penalidade_aresta + 0.1 * len(cor_count) + 0.5 * desequilibrio

    def fitness(self, cromossomo):
        return -1 * self.FO(cromossomo)

    def construtivo_aleatorio(self):
        return [[randint(1, self.grafo.V) for _ in range(self.grafo.V)] for _ in range(self.tamanho_populacao)]

    def set_elite(self, populacao: list):
        populacao.sort(key=lambda c: self.FO(c))
        return populacao[:self.len_elites]

    def crossover(self, pai1, pai2):
        ponto_corte = randint(1, len(pai1) - 1)
        return pai1[:ponto_corte] + pai2[ponto_corte:]

    def mutacao(self, cromossomo):
        idx = randint(0, len(cromossomo) - 1)
        cromossomo[idx] = randint(1, self.grafo.V)

    def selecionar_pais_torneio(self, populacao, fitness_populacao):
        pais = []
        for _ in range(len(populacao) // 2):
            selecionados = sample(list(zip(populacao, fitness_populacao)), 2)
            pais.append(max(selecionados, key=lambda x: x[1])[0])

        if len(pais) % 2 != 0:
            pais.pop()
        return pais

    def selecionar_pais_roleta(self, populacao, fitness_populacao):
        soma_fitness = sum(fitness_populacao)
        if soma_fitness == 0:
            probabilidades = [1 / len(fitness_populacao) for _ in fitness_populacao]
        else:
            probabilidades = [fitness / soma_fitness for fitness in fitness_populacao]

        pais = []
        for _ in range(len(populacao) // 2):
            r = uniform(0, 1)
            acumulador = 0
            for i, prob in enumerate(probabilidades):
                acumulador += prob
                if r <= acumulador:
                    pais.append(populacao[i])
                    break
        return pais

    def execute(self, geracoes=100, ordem_selecao=("torneio", "roleta")):
        populacao = self.construtivo_aleatorio()
        for geracao in range(geracoes):
            fitness_populacao = [self.fitness(cromossomo) for cromossomo in populacao]
            elites = self.set_elite(populacao)

            if ordem_selecao[0] == "torneio":
                melhores_pais = self.selecionar_pais_torneio(populacao, fitness_populacao)
                fitness_pais = [self.fitness(cromossomo) for cromossomo in melhores_pais]
                melhores_pais = self.selecionar_pais_roleta(melhores_pais, fitness_pais)
            else:
                melhores_pais = self.selecionar_pais_roleta(populacao, fitness_populacao)
                fitness_pais = [self.fitness(cromossomo) for cromossomo in melhores_pais]
                melhores_pais = self.selecionar_pais_torneio(melhores_pais, fitness_pais)

            # Garantir que o número de pais seja par
            if len(melhores_pais) % 2 != 0:
                melhores_pais.pop()

            nova_populacao = elites.copy()
            for i in range(0, len(melhores_pais), 2):
                pai1, pai2 = melhores_pais[i], melhores_pais[i + 1]
                filho1 = self.crossover(pai1, pai2)
                filho2 = self.crossover(pai2, pai1)
                self.mutacao(filho1)
                self.mutacao(filho2)
                nova_populacao.extend([filho1, filho2])

            populacao = nova_populacao[:self.tamanho_populacao]

        melhor_individuo = min(populacao, key=lambda cromossomo: self.FO(cromossomo))
        return melhor_individuo



def comparar_metodos(ag, geracoes=10, ordens=[("torneio", "roleta"), ("roleta", "torneio")]):
    resultados = []

    for ordem in ordens:
        start_time = time.time()
        melhor_colocacao = ag.execute(geracoes=geracoes, ordem_selecao=ordem)
        elapsed_time = time.time() - start_time
        melhor_fitness = ag.fitness(melhor_colocacao)

        resultados.append({
            "ordem": " -> ".join(ordem),
            "tempo": elapsed_time,
            "fitness": melhor_fitness
        })

    return resultados

def plotar_resultados(resultados):
    ordens = [resultado["ordem"] for resultado in resultados]
    tempos = [resultado["tempo"] for resultado in resultados]
    fitness = [resultado["fitness"] for resultado in resultados]

    # Gráfico de tempos de execução
    plt.figure(figsize=(12, 6))
    plt.bar(ordens, tempos, color='blue', alpha=0.7)
    plt.title("Comparação de Tempos de Execução")
    plt.ylabel("Tempo de Execução (s)")
    plt.xlabel("Ordem dos Métodos de Seleção")
    plt.show()

    # Gráfico de fitness das soluções
    plt.figure(figsize=(12, 6))
    plt.bar(ordens, fitness, color='green', alpha=0.7)
    plt.title("Comparação de Fitness das Melhores Soluções")
    plt.ylabel("Fitness (quanto maior, melhor)")
    plt.xlabel("Ordem dos Métodos de Seleção")
    plt.show()

if __name__ == "__main__":
    v, a, arestas = read_DIMACS('./input-data/S10.txt')
    g = Grafo(v)
    [g.add_aresta(aresta[0], aresta[1], 0.0) for aresta in arestas]

    ag = AlgoritmoGenetico(g)

    resultados = comparar_metodos(ag, geracoes=10, ordens=[("torneio", "roleta"), ("roleta", "torneio")])
    plotar_resultados(resultados)'''
