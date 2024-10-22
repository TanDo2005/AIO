import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)



def load_data_from_file(fileName = "d:/AIO/Module 4/Code/Week3/advertising.csv"):
    data = np.genfromtxt(fileName, dtype= None, delimiter= ',', skip_header= 1)
    features_X = data[:, :3]
    sales_Y = data[:, 3]

    features_X = np.hstack((np.ones((features_X.shape[0], 1)), features_X))

    return features_X, sales_Y


# Question 2, 3

features_X, sales_Y = load_data_from_file()

# print(features_X[:5, :])

# print(sales_Y.shape)

def create_individual(n= 4, bound= 10):
    individual = [random.uniform(-bound / 2, bound / 2) for _ in range(n)]
    return individual

individual = create_individual()

def compute_loss(individual):
    theta = np.array(individual)
    y_hat = features_X.dot(theta)
    loss = np.multiply((y_hat - sales_Y), (y_hat - sales_Y)).mean()

    return loss

def compute_fitness(individual):
    loss = compute_loss(individual)
    fitness_value = 1 / (loss + 1)

    return fitness_value

# Question 4

# features_X, sales_Y = load_data_from_file()
# individual = [4.09, 4.82, 3.10, 4.02]
# fitness_score = compute_fitness(individual)

# print(fitness_score)


def crossover(individual1, individual2, crossover_rate = 0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    for i in range(len(individual1)):

        if random.random() < crossover_rate:
            individual1_new[i], individual2_new[i] = individual2[i], individual1[i]
    
    return individual1_new, individual2_new

# Question 5

# individual1 = [4.09 , 4.82 , 3.10 , 4.02]
# individual2 = [3.44 , 2.57 , -0.79 , -2.41]
# individual1 , individual2 = crossover ( individual1 , individual2 , 2.0)
# print (" individual1 : " , individual1 )
# print (" individual2 : " , individual2 )


def mutate(individual, mutation_rate= 0.05):
    individual_m = individual.copy()
    
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual_m[i] = random.uniform(-5, 5)

    return individual_m

# Question 6

# before_individual = [4.09 , 4.82 , 3.10 , 4.02]
# after_individual = mutate ( individual , mutation_rate = 2.0)
# print ( before_individual == after_individual )

def initializePopulation(m):
    population = [create_individual() for _ in range(m)]

    return population

def selection(sorted_old_population, m = 100):
    index1 = random.randint(0, m - 1)
    while True:
        index2 = random.randint(0, m - 1)
        if (index2 != index1):
            break

    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s

def create_new_population(old_population, elitism= 2, gen= 1):
    m = len(old_population)
    sorted_population = sorted(old_population, key= compute_fitness)

    if gen % 1 == 0:
        print("Best loss:", compute_loss(sorted_population[m - 1]), "with chromosome: ", sorted_population[m - 1])
    
    new_population = []

    while len(new_population) < m - elitism:
        #selection
        individual1 = selection(sorted_population, m)
        individual2 = selection(sorted_population, m)

        #crossover
        individual1, individual2 = crossover(individual1, individual2)

        #mutation
        individual1 = mutate(individual1)
        individual2 = mutate(individual2)

        new_population.append(individual1)
        new_population.append(individual2)

    for ind in sorted_population[m - elitism:]:
        new_population.append(ind)

    new_population = sorted(new_population, key= compute_fitness)

    return new_population, compute_loss(sorted_population[m - 1])

# Question 7 

# individual1 = [4.09 , 4.82 , 3.10 , 4.02]
# individual2 = [3.44 , 2.57 , -0.79 , -2.41]
# old_population = [ individual1 , individual2 ]
# new_population , _ = create_new_population ( old_population , elitism =2 , gen =1)

def run_GA():
    n_generation = 100
    m = 600
    features_X, sales_Y = load_data_from_file()
    population = initializePopulation(m)
    losses_list = []

    for i in range(n_generation):
        best_loss = compute_loss(sorted(population, key=compute_fitness)[-1])
        losses_list.append(best_loss)

        population, _ = create_new_population(population)

    return losses_list

# losses_list = run_GA()

def visualize_loss(losses_list):
    x_axis = list(range(100))
    plt.plot(x_axis, losses_list)
    plt.show()
    plt.xlabel('Generation')
    plt.ylabel('Loss')

# losses_list = run_GA()
# visualize_loss(losses_list)


population = initializePopulation(600)
sorted_population = sorted(population, key=compute_fitness)
print(sorted_population[-1])
theta = np.array(sorted_population[-1])

estimated_prices = []
samples = [i for i in range(len(features_X))]
for feature in features_X:
    estimated_price = sum(c*x for x, c in zip(feature, theta))
    estimated_prices.append(estimated_price)
fig, ax = plt.subplots(figsize=(10, 6))
# plt.plot(prices, c='green')
# plt.plot(estimated_prices, c='red')
plt.xlabel('Samples')
plt.ylabel('Price')
plt.scatter(samples, sales_Y, c='green', label='Real Prices')
plt.scatter(samples, estimated_prices, c='blue', label='Estimated Prices')
plt.legend()
plt.show()