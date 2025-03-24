import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os

# Function to load input data from a file
def load_input_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        n, capacity = map(int, lines[0].split())
        items = [tuple(map(int, line.split())) for line in lines[1:]]
    return n, capacity, items

# Function to evaluate the fitness of a solution
def evaluate(solution, items, capacity):
    total_weight = sum(items[i][1] for i in range(len(solution)) if solution[i] == 1)
    total_value = sum(items[i][0] for i in range(len(solution)) if solution[i] == 1)
    if total_weight > capacity:
        return 0  # Penalize overweight solutions more severely
    return total_value

# Function to repair solutions that exceed the capacity
def repair_solution(solution, items, capacity):
    total_weight = sum(items[i][1] for i in range(len(solution)) if solution[i] == 1)
    while total_weight > capacity:
        ones = [i for i in range(len(solution)) if solution[i] == 1]
        if not ones:
            break
        remove_idx = random.choice(ones)
        solution[remove_idx] = 0
        total_weight -= items[remove_idx][1]
    return solution

# Function to initialize the population with all individuals being 0
def initialize_population(pop_size, items, capacity):
    return np.zeros((pop_size, len(items)), dtype=int)
    

# Function to select parents using tournament selection
def tournament_selection(population, fitnesses, num_parents, tournament_size=3):
    selected = []
    for _ in range(num_parents):
        participants = random.sample(list(enumerate(fitnesses)), tournament_size)
        best = max(participants, key=lambda x: x[1])[0]
        selected.append(population[best])
    return selected

# Function to perform crossover between two parents
def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    return child1, child2

# Function to mutate a solution
def mutate(solution, mutation_rate):
    return [(1 - gene) if random.random() < mutation_rate else gene for gene in solution]

# Main genetic algorithm function
def genetic_algorithm(items, capacity, pop_size=500, generations=200, crossover_rate=0.8, initial_mutation_rate=0.0008, mutation_decay=0.00004):
    n = len(items)
    population = initialize_population(pop_size, items, capacity)
    best_fitnesses, avg_fitnesses = [], []
    mutation_rate = initial_mutation_rate

    for gen in range(generations):
        fitnesses = [evaluate(ind, items, capacity) for ind in population]
        best_fit = max(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)
        best_fitnesses.append(best_fit)
        avg_fitnesses.append(avg_fit)

        if gen % 50 == 0:
            print(f"Generation {gen}: best = {best_fit}, avg = {avg_fit:.2f}")

        elite_count = max(1, pop_size // 10)
        elites = [population[i] for i in np.argsort(fitnesses)[-elite_count:]]

        parents = tournament_selection(population, fitnesses, pop_size // 2, tournament_size=5)

        next_population = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = crossover(parents[i], parents[i + 1], crossover_rate)
            mutated_child1 = mutate(child1, mutation_rate)
            mutated_child2 = mutate(child2, mutation_rate)
            next_population.append(repair_solution(mutated_child1, items, capacity))
            next_population.append(repair_solution(mutated_child2, items, capacity))

        # Handle odd parent
        if len(parents) % 2 == 1:
            mutated_child = mutate(parents[-1], mutation_rate)
            next_population.append(repair_solution(mutated_child, items, capacity))

        population = elites + next_population
        population = population[:pop_size]

    best_solution = max(population, key=lambda ind: evaluate(ind, items, capacity))
    best_value = evaluate(best_solution, items, capacity)

    plt.figure()
    plt.plot(best_fitnesses, label="Best Fitness")
    plt.plot(avg_fitnesses, label="Average Fitness")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/fitness_plot_{os.path.basename(sys.argv[1]).replace(".txt","")}.png')

    return best_solution, best_value

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    n, capacity, items = load_input_file(input_file)

    best_solution, best_value = genetic_algorithm(items, capacity)
    print(f"Best solution value for {input_file}: {best_value}")
