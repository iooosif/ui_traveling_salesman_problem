import random

import matplotlib.pyplot as plt

import numpy as np

# Сreate a list of cities with random coordinates



def generate_cities(n, max_x=200, max_y=200):
    return [(random.randint(0, max_x), random.randint(0, max_y)) for i in range(n)]


def euclidean_dist(city1, city2):
    return ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2) ** 0.5


#  calculate the total length of the route
def calculate_distance(route, cities):
    total_distance = 0
    for i in range(len(route)):
        total_distance += euclidean_dist(cities[route[i - 1]], cities[route[i]])
    return total_distance


def fitness(path, cities):
    # reciprocal of distance
    return 1 / calculate_distance(path, cities)


# Initialization of the population
def initialize_pop(pop_size, num_cities):
    population = []
    for i in range(pop_size):
        individual = list(np.random.permutation(num_cities))
        population.append(individual)
    return population


# Select random individuals, evaluate fitness, select tournament winner,
# individual with highest fitness value is considered tournament winner
# and returned as one of the parents
def tournament_selection(population, cities, k=6):
    selected = random.sample(population, k)
    selected_fitness = [fitness(ind, cities) for ind in selected]
    return selected[np.argmax(selected_fitness)]


# roulette wheel selection
#def roulette_wheel_selection(population, cities):
   # fitness_scores = [1/fitness(ind, cities) for ind in population]

   #total_fitness = sum(fitness_scores)

    # Transforming Fitness into Probabilities
   # probabilities = [fit / total_fitness for fit in fitness_scores]
   # print(sum(probabilities))
   # cumulative_prob = np.cumsum(probabilities)
   # r = random.random()

  #  for i, prob in enumerate(cumulative_prob):
       # if r < prob:
        #    return population[i]


# Selecting a random subsegment from the first parent, the subsegment of the route of the first parent parent1[start:end]
# is copied to the same place in the descendant. Filling in the remaining cities from the second parent,
# checking that there are no duplicate cities
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size

    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]

    idx = end
    for gene in parent2:
        if gene not in child:
            if idx >= size:
                idx = 0
            child[idx] = gene
            idx += 1

    return child


# Mutation: inversion of the route section; the cities in the section are rearranged in the opposite order
def mutation(individual):
    start, end = sorted(random.sample(range(len(individual)), 2))
    individual[start:end] = reversed(individual[start:end])
    return individual


# genetic algorithm
def genetic_algorithm(cities, pop_size=100, generations=300, mutation_rate=0.2):
    num_cities = len(cities)
    population = initialize_pop(pop_size, num_cities)
    # to track the best distance in each generation.
    best_distances = []

    for generation in range(generations):
        new_population = []
        for _ in range(pop_size // 2):
            # For each new individual, two parents are selected.
            parent1 = tournament_selection(population, cities)
            parent2 = tournament_selection(population, cities)
           # roulette_wheel_selection(population, cities)
           # parent1 = roulette_wheel_selection(population, cities)
           # parent2 = roulette_wheel_selection(population, cities)
            # Crossover is applied to parents to create offspring
            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)
            # the offspring may be mutated
            if random.random() < mutation_rate:
                child1 = mutation(child1)
            if random.random() < mutation_rate:
                child2 = mutation(child2)

            new_population.append(child1)
            new_population.append(child2)
        # After the formation of a new population, the old one is replaced by a new one.
        population = new_population
        #Finding the best solution in a generation
        best_individual = max(population, key=lambda ind: fitness(ind, cities))
        best_distance = calculate_distance(best_individual, cities)
        best_distances.append(best_distance)
        print(f"Generation {generation + 1}: Best Distance = {best_distance:.2f} km")

    return best_individual, best_distances


#Graf with best route
def plot_route(cities, best_route, title):
    plt.figure(figsize=(10, 6))
    # Connecting cities
    for i in range(len(best_route)):
        plt.plot([cities[best_route[i - 1]][0], cities[best_route[i]][0]],
                 [cities[best_route[i - 1]][1], cities[best_route[i]][1]], 'b')  # Route lines
    x, y = zip(*cities)
    plt.scatter(x, y, c='r', marker='o')  # Cities as points
    # City numbers
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


#graph of the change of the best distance
def plot_evolution(best_distances):
    plt.figure(figsize=(10, 6))
    plt.plot(best_distances, color='r')
    plt.title("Best Distance Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Distance (km)")
    # Adding a Grid
    plt.grid(True)
    plt.show()


count = int(input('enter cities count:'))
cities = generate_cities(count)

#cities = [(85, 69), (137, 102), (15, 47), (158, 19), (153, 85), (146, 190), (133, 173), (169, 46), (6, 171), (129, 157),
   #      (103, 35), (107, 193), (139, 186), (196, 100), (81, 22), (151, 174), (197, 65), (151, 90), (143, 18),
    #      (75, 173), (157, 48), (35, 110), (50, 193), (171, 16), (21, 106), (166, 111), (179, 196), (112, 125),
     #    (109, 37), (160, 132)]
# Launching the genetic algorithm
best_path, best_distances = genetic_algorithm(cities)

# Displaying the best results
print("Best path:", [int(i) for i in best_path])
print("Best total distance:", calculate_distance(best_path, cities))

# Showing the best route
plot_route(cities, best_path, f"Final Route with Distance: {calculate_distance(best_path, cities):.2f}")

# Displaying a graph of population change
plot_evolution(best_distances)
