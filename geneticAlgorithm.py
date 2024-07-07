'''
Worksheet 1 - Modeling and Simulation
Date: 06/2024

Implementation of a genetic algorithm to find the maximum value
of the function f(x) = x * sin(10πx) + 1 within the interval [0,1].
'''

import random as r
import math
import matplotlib.pyplot as plt

# Genetic algorithm parameters
GENERATIONS = 10
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
GENES = 8

'''
Define the function to be maximized.

Parameters:
    x (float): The input value
    
Returns:
    float: The output value of the function
'''
def function(x):
    return x * math.sin(10 * math.pi * x) + 1

'''
Generate the initial population of chromosomes.

Returns:
    list: The initial population of chromosomes
'''
def initial_population():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = ''
        for _ in range(GENES):
            chromosome += str(r.randint(0, 1))
        population.append(chromosome)
    return population

'''
Calculate the fitness of a chromosome.

Parameters:
    chromosome (str): The chromosome to evaluate
    
Returns:
    float: The fitness value of the chromosome
'''
def fitness(chromosome):
    # Convert the chromosome to a decimal number and normalize
    x = int(chromosome, 2) / (2**GENES)
    return function(x)

'''
Select the best individuals from the population using tournament selection.

Parameters:
    population (list): The population of chromosomes
    tournament_size (int): The number of individuals to compete in each tournament
    
Returns:
    list: The selected individuals
'''
def tournament_selection(population, tournament_size=3):
    selected = []
    for _ in range(POPULATION_SIZE):
        # Select random individuals from the population to compete
        tournament = r.sample(population, tournament_size)
        best_in_tournament = max(tournament, key=fitness)
        selected.append(best_in_tournament)
    return selected

'''
Perform crossover between two parent chromosomes.

Parameters:
    parent1 (str): The first parent chromosome
    parent2 (str): The second parent chromosome
    
Returns:
    tuple: The two child chromosomes
'''
def crossover(parent1, parent2):
    # Random crossover point
    crossover_point = r.randint(1, GENES - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

'''
Mutate a chromosome by flipping bits with a certain probability.

Parameters:
    chromosome (str): The chromosome to mutate
    
Returns:
    str: The mutated chromosome
'''
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
    max_fitness_history = []
    avg_fitness_history = []

    for generation in range(GENERATIONS):
        selected = tournament_selection(population)
        new_population = []
        for parents in selected:
            child1, child2 = crossover(parents, parents)
            new_population.append(mutation(child1))
            new_population.append(mutation(child2))

         # Elitisim
        if best_overall is None or fitness(max(new_population, key=fitness)) > fitness(best_overall):
            best_overall = max(new_population, key=fitness)

        new_population.append(best_overall)
        # Adjust the population size
        population = new_population[:POPULATION_SIZE]

        best_x = int(best_overall, 2) / (2**GENES)
        
        # Calculate and store the maximum and average fitness of the population
        current_fitness = [fitness(chromosome) for chromosome in population]
        max_fitness_history.append(max(current_fitness))
        avg_fitness_history.append(sum(current_fitness) / len(current_fitness))

        print(f'\nGeneration {generation + 1}')
        print(f'Best chromosome: {best_overall}')
        print(f'Best x: {best_x}')
        print(f'Best f(x): {fitness(best_overall)}')

    plot_fitness(max_fitness_history, avg_fitness_history)

def plot_fitness(max_fitness, avg_fitness):
    plt.figure(figsize=(10, 5))
    plt.plot(max_fitness, label='Max Fitness')
    plt.plot(avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
