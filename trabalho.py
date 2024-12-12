from random import choices, randint, random, sample
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ReadData import read_DIMACS, instancias
from representacao import Grafo, Individuo

def printGrafo(v: int, a: int, title: str, colors: list = ['white']):
    G = nx.Graph()
    [G.add_node(i+1) for i in range(v)]
    [G.add_edge(arestas[i][0], arestas[i][1]) for i in range(a)]

    plt.title(title)
    nx.draw(G, with_labels=True, node_color=colors, node_size=450, edgecolors='black', linewidths=1)
    plt.show()

class AlgoritmoGenetico:
    def __init__(self, grafo: Grafo):
        self.grafo = grafo
        self.tamanho_populacao = 2 * grafo.V
        self.len_elites = 0.2 * self.tamanho_populacao
        self.len_nao_elites = self.tamanho_populacao - self.len_elites

    def FO(self, cromossomo):
        penalidade_aresta = 0
        for a in self.grafo.arestas:
            if cromossomo[a.origem - 1] == cromossomo[a.destino - 1]:
                penalidade_aresta += 1

        qtd_cores = len(set(cromossomo))

        cor_count = Counter(cromossomo)
        floor_val = self.grafo.V // qtd_cores
        ceil_val = floor_val + (1 if self.grafo.V % qtd_cores != 0 else 0)

        desequilibrio = 0
        for ccount in cor_count.values():
            if ccount < floor_val:
                desequilibrio += (floor_val - ccount) ** 2
            elif ccount > ceil_val:
                desequilibrio += (ccount - ceil_val) ** 2

        return {"fo": float(penalidade_aresta * 1000 + qtd_cores * 100 + desequilibrio * 10), "p": penalidade_aresta, "c": qtd_cores, "d": desequilibrio}

    def fitness(self, cromossomo):
        obj = self.FO(cromossomo)
        obj['fo'] *= -1
        return obj

    def construtivo_aleatorio(self):
        populacao = []
        for _ in range(self.tamanho_populacao):
            cromossomo = [int(random()*10) for _ in range(self.grafo.V)]
            f = self.fitness(cromossomo)
            populacao.append(Individuo(cromossomo, f['fo'], f['c'], f['p'], f['d']))
        return populacao

    def set_elite(self, populacao: list):
        fitness_ordenado = sorted(populacao, key=lambda i: i.value, reverse=True)
        melhores = fitness_ordenado[:int(self.len_elites)]
        piores = fitness_ordenado[int(self.len_elites):]
        return melhores, piores

    def crossover(self, pai1, pai2):
        ponto_corte = randint(1, len(pai1.cromossomo) - 1)
        filho = pai1.cromossomo[:ponto_corte] + pai2.cromossomo[ponto_corte:]
        f = self.fitness(filho)
        return Individuo(filho, f['fo'], f['c'], f['p'], f['d'])

    def mutacao(self, individuo: Individuo):
        gene_aleatorio = randint(0, len(individuo.cromossomo) - 1)
        individuo.cromossomo[gene_aleatorio] = randint(1, len(individuo.cromossomo) - 1)
        f = self.fitness(individuo.cromossomo)
        individuo.value = f['fo']
        individuo.qtd_cores = f['c']
        individuo.rest_aresta = f['p']
        individuo.desequilibrio = f['d']

    def torneio_numpy(self, populacao, N, q = 2):
        fitness = np.array([ind.value for ind in populacao])
        torneios = np.random.choice(len(populacao), size=(N, q), replace=True)
        vencedores_indices = torneios[np.arange(N), np.argmax(fitness[torneios], axis=1)]
        return [populacao[idx] for idx in vencedores_indices]


    def roleta_numpy(self, populacao, m: int):
        fitness = np.array([-ind.value for ind in populacao])

        if np.min(fitness) < 0:
            fitness -= np.min(fitness)

        if np.sum(fitness) == 0:
            probabilities = np.full(len(populacao), 1 / len(populacao))
        else:
            probabilities = fitness / np.sum(fitness)  # Probabilidades proporcionais ao fitness

        selecionados_indices = np.random.choice(len(populacao), size=m, p=probabilities, replace=True)

        return [populacao[i] for i in selecionados_indices]


    def execute(self, geracoes=1000):
        populacao = self.construtivo_aleatorio()
        melhor_inicial = max(populacao, key=lambda ind: ind.value)
        # resultado.write('Melhor cromossomo do construtivo:', melhor_inicial.cromossomo)
        resultado.write(f'\nQtd cores: {melhor_inicial.qtd_cores}')
        resultado.write(f'\nPenalidade restrição de arestas: {melhor_inicial.rest_aresta}')
        resultado.write(f'\nDesequilibrio de cores: {melhor_inicial.desequilibrio}')

        for geracao in range(geracoes):
            elites, nao_elites = self.set_elite(populacao)
            nova_populacao = elites.copy()
            novos_individuos = []

            while len(novos_individuos) < self.len_nao_elites:
                probabilidade_cruz = random()
                probabilidade_mut = random()
                if probabilidade_cruz < 0.94:
                    # pais = self.roleta_numpy(populacao, 2)
                    pais = self.torneio_numpy(populacao, 2)
                    filho1 = self.crossover(pais[0], pais[1])
                    filho2 = self.crossover(pais[1], pais[0])
                    novos_individuos.append(filho1)
                    novos_individuos.append(filho2)
                
                if probabilidade_mut <= 0.05:
                    aleatorio = sample(nao_elites, 1)
                    self.mutacao(aleatorio[0])

            uniao = populacao + novos_individuos
            # escolhidos = self.roleta_numpy(uniao, len(nao_elites))
            escolhidos = self.torneio_numpy(uniao, len(nao_elites))
            populacao.clear()
            populacao.extend(nova_populacao + escolhidos)

        melhor_individuo = max(populacao, key=lambda ind: ind.value)
        return melhor_individuo

def main():
    g = Grafo(v)
    [g.add_aresta(arestas[i][0], arestas[i][1], 0.0) for i in range(a)]
    resultado.write(str(g))
    # printGrafo(v, a, str(g))

    ag = AlgoritmoGenetico(g)
    ag_inicio = datetime.now()
    melhor_colocacao: Individuo = ag.execute(geracoes=500)
    ag_fim = datetime.now()
    # resultado.write('Melhor cromossomo final: ', melhor_colocacao.cromossomo)
    resultado.write(f'\nQtd cores: {melhor_colocacao.qtd_cores}')
    resultado.write(f'\nPenalidade restrição de arestas: { melhor_colocacao.rest_aresta}')
    resultado.write(f'\nDesequilibrio de cores: {melhor_colocacao.desequilibrio}')
    if TARGET.get(nome) is None:
        resultado.write(f'\nGAP: -- não há target na literatura para esta instância')
    else:
        resultado.write(f'\nGAP: {((melhor_colocacao.qtd_cores + melhor_colocacao.rest_aresta + melhor_colocacao.desequilibrio)- TARGET.get(nome)) / TARGET.get(nome)*100}')
    resultado.write(f'\nTempo de execução: {ag_fim - ag_inicio}\n')

    # printGrafo(v, a, str(g) + " \nFitness da solução: " + str(melhor_colocacao.value), melhor_colocacao.cromossomo)

if __name__ == "__main__":
    TARGET = {
        'S09.txt': None,
        '1-FullIns_3.txt': 4,
        'S20.txt': None,
        'ale70_10_1.col': 4,
        'ale50_10_5.col': 3,
        'K7_2.txt': 6,
        'K5_2.txt': 3,
        'ale70_10_4.col': 4,
        'ale60_10_4.col': 4,
        'ale50_10_1.col': 4,
        'ale50_10_4.col': 4,
        'ale60_10_1.col': 4,
        'K7_3.txt': 3,
        'ale50_10_3.col': 4,
        'ale50_10_2.col': 3,
        '2-FullIns_3.txt': 5,
        'ale60_10_3.col': 4,
        'ale70_10_2.col': 4,
        'ale60_10_2.col': 4,
        'ale70_10_5.col': 4,
        'ale60_10_5.col': 4,
        'myciel4.txt': 5,
        'myciel3.txt': 4,
        'S10.txt': None,
        'ale70_10_3.col': 4,
    }

    for nome in instancias():
        v, a, arestas = read_DIMACS('./input-data/' + nome)
        with open(f"result_500geracoes_TT_{nome}", "w") as resultado:
            resultado.write(f'Instância: {nome} \n')
            main()