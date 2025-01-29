import random
from itertools import combinations
import matplotlib.pyplot as plt

#Сreate a list of cities with random coordinates
def generate_cities(n, max_x=200, max_y=200):
    return [(random.randint(0, max_x), random.randint(0, max_y)) for i in range(n)]

def euclidean_dist(city1, city2):
    return ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)**0.5

# Generating neighbors (by swapping two cities in a route)
def generate_neighbors(route):
    neighbors = []
    for i, j in combinations(range(len(route)), 2):
        neighbor = route[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbors.append(neighbor)
    return neighbors

# Calculating the total length of a route
def calculate_distance(route, cities):
    total_distance = 0
    for i in range(len(route)):
        total_distance += euclidean_dist(cities[route[i - 1]], cities[route[i]])
    return total_distance


# Graf with best route
def plot_route(cities, best_route, title):
    plt.figure(figsize=(10, 6))
    # Connecting cities
    for i in range(len(best_route)):
        plt.plot([cities[best_route[i - 1]][0], cities[best_route[i]][0]], [cities[best_route[i - 1]][1], cities[best_route[i]][1]], 'r')  # Route lines
    x, y = zip(*cities)
    plt.scatter(x, y, c='black', marker='o')  # Cities as points
    # City numbers
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
#graph of the change of the best distance
def plot_performance(best_distances):
    plt.figure(figsize=(10, 6, ), )
    plt.plot(best_distances, marker='o')
    plt.title("Changing over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    #Adding a Grid
    plt.grid(True)
    plt.show()

def tabu_search(cities, tabu_size):
    n = len(cities)
    best_route = list(range(n))

    # Initial random assignment of cities
    random.shuffle(best_route)
    best_distance = calculate_distance(best_route, cities)

    current_route = best_route[:]
    current_distance = best_distance

    tabu_list = [current_route[:]]
    best_distances = [best_distance]  # For performance tracking

    # for iteration, while best distance changes:
    iteration = 0
    while iteration < 150:
        # neighbors generation
        #Generate neighboring routes obtained by swapping two cities in the current route.
        neighbors = generate_neighbors(current_route)

        # Finding the best neighbor outside of tabu list
        best_neighbor = None
        best_neighbor_distance = 99999999.9

        for neighbor in neighbors:
            if neighbor not in tabu_list:
                distance = calculate_distance(neighbor, cities)
                if distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = distance


        # Updating the current solution
        if best_neighbor is not None:
            current_route = best_neighbor
            current_distance = best_neighbor_distance

        # Adding current solution to tabu list
        tabu_list.append(current_route[:])
        if len(tabu_list) > tabu_size:
            # Delete the oldest element if the size is exceeded
            tabu_list.pop(0)

        # Updating the best solution
        if current_distance < best_distance:
            best_route = current_route[:]
            best_distance = current_distance

        best_distances.append(best_distance)
        #if the length no longer changes - we exit
        if (best_distances[-1] == best_distances[-2] == best_distances[-3] == best_distances[-4] == best_distances[-5]) and iteration >=100:
            break
        print(f"Iteration {iteration + 1}: best distance {best_distance}")
        iteration += 1

    return best_route, best_distance, best_distances


count = int(input('enter cities count:'))
cities = generate_cities(count)
#cities = [(85, 69), (137, 102), (15, 47), (158, 19), (153, 85), (146, 190), (133, 173), (169, 46), (6, 171), (129, 157), (103, 35), (107, 193), (139, 186), (196, 100), (81, 22), (151, 174), (197, 65), (151, 90), (143, 18), (75, 173), (157, 48), (35, 110), (50, 193), (171, 16), (21, 106), (166, 111), (179, 196), (112, 125), (109, 37), (160, 132)]
#print(cities)
# Tabu Search Options
tabu_size = 70


# Launching the algorithm
best_route, best_distance, best_distances = tabu_search(cities, tabu_size)



print(f"Best route: {best_route}")
print(f"Best distance: {best_distance}")
# Plotting the final route
plot_route(cities, best_route, f"Final Route with Distance: {best_distance:.2f}")


# Plotting performance graph
plot_performance(best_distances)
