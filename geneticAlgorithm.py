'''
Worksheet 1 - Modeling and Simulation
Date: 06/2024

Implementation of a genetic algorithm to find the maximum value
of the function f(x) = x * sin(10πx) + 1 within the interval [0,1].
'''

import random as r
import math

GENERATIONS = 10
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
GENES = 8

def function(x):
    return x * math.sin(10 * math.pi * x) + 1

def initial_population():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = ''
        for _ in range(GENES):
            chromosome += str(r.randint(0, 1))
        population.append(chromosome)
    return population

def fitness(chromosome):
    # Convert the chromosome to a decimal number and normalize
    x = int(chromosome, 2) / (2**GENES)
    return function(x)

def tournament_selection(population, tournament_size=3):
    selected = []
    for _ in range(POPULATION_SIZE):
        # Select random individuals from the population to compete
        tournament = r.sample(population, tournament_size)
        best_in_tournament = max(tournament, key=fitness)
        selected.append(best_in_tournament)
    return selected

def crossover(parent1, parent2):
    # Random crossover point
    crossover_point = r.randint(1, GENES - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutation(chromosome):
    mutated_chromosome = ''
    for gene in chromosome:
        if r.random() < MUTATION_RATE:
            # Flip the bit
            mutated_chromosome += '1' if gene == '0' else '0'
        else:
            mutated_chromosome += gene
    return mutated_chromosome

def main():
    population = initial_population()
    # Best chromosome
    best_overall = None

    for generation in range(GENERATIONS):
        selected = tournament_selection(population)

        new_population = []
        for parents in selected:
            child1, child2 = crossover(parents, parents)
            new_population.append(mutation(child1))
            new_population.append(mutation(child2))

        # Elitisim
        best_chromosome = max(population, key=fitness)
        if best_overall is None or fitness(best_chromosome) > fitness(best_overall):
            best_overall = best_chromosome

        new_population.append(best_overall)
        # Adjust the population size
        population = new_population[:POPULATION_SIZE]

        best_x = int(best_overall, 2) / (2**GENES)
        print(f'\nGeneration {generation + 1}')
        print(f'Best chromosome: {best_overall}')
        print(f'Best x: {best_x}')
        print(f'Best f(x): {function(best_x)}')

if __name__ == '__main__':
    main()

