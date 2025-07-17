import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import random

# Load and preprocess dataset
data = pd.read_csv('Rainfall.csv')

# Map labels
data['rainfall'] = data['rainfall'].map({'yes': 1, 'no': 0})

# Drop unnecessary columns  a
if 'day' in data.columns:
    data.drop('day', axis=1, inplace=True)

# Convert columns to numeric
for col in data.columns:
    if col != 'rainfall':
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Class balance check
print("Class distribution:", data['rainfall'].value_counts())

# Features and labels
X = data.drop('rainfall', axis=1)
y = data['rainfall']

if y.nunique() < 2:
    raise ValueError("Target column 'rainfall' has only one class. Cannot train.")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Genetic Algorithm Parameters
POP_SIZE = 15
GENERATIONS = 20

def create_individual():
    layers = random.choice([1, 2])
    if layers == 1:
        hidden = (random.randint(10, 60),)
    else:
        hidden = (random.randint(10, 60), random.randint(10, 60))
    learning_rate = random.uniform(0.0005, 0.05)
    return {'hidden_layer_sizes': hidden, 'learning_rate_init': learning_rate}

def fitness(individual):
    try:
        clf = MLPClassifier(hidden_layer_sizes=individual['hidden_layer_sizes'],
                            learning_rate_init=individual['learning_rate_init'],
                            max_iter=500,
                            random_state=42)
        score = cross_val_score(clf, X_train, y_train, cv=4, scoring='accuracy')
        return score.mean()
    except:
        return 0

def crossover(p1, p2):
    child = {}
    child['hidden_layer_sizes'] = random.choice([p1['hidden_layer_sizes'], p2['hidden_layer_sizes']])
    child['learning_rate_init'] = (p1['learning_rate_init'] + p2['learning_rate_init']) / 2
    return child

def mutate(individual, rate=0.2):
    if random.random() < rate:
        layers = len(individual['hidden_layer_sizes'])
        if layers == 1:
            individual['hidden_layer_sizes'] = (random.randint(10, 60),)
        else:
            individual['hidden_layer_sizes'] = (random.randint(10, 60), random.randint(10, 60))
    if random.random() < rate:
        delta = random.uniform(-0.005, 0.005)
        new_lr = individual['learning_rate_init'] + delta
        individual['learning_rate_init'] = min(max(new_lr, 0.0005), 0.05)
    return individual

# GA Optimization
population = [create_individual() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    fitnesses = [fitness(ind) for ind in population]
    print(f"Generation {gen+1} - Best Accuracy: {max(fitnesses):.4f}")
    sorted_population = [ind for _, ind in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]
    population = sorted_population[:POP_SIZE//2]  # Keep top half
    children = []
    while len(children) + len(population) < POP_SIZE:
        p1, p2 = random.sample(population, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        children.append(child)
    population.extend(children)

# Best individual
best_individual = max(population, key=fitness)
print(f"\n🔥 Best Hyperparameters: {best_individual}")

# Train final model
final_model = MLPClassifier(hidden_layer_sizes=best_individual['hidden_layer_sizes'],
                            learning_rate_init=best_individual['learning_rate_init'],
                            max_iter=500,
                            random_state=42)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Final Test Accuracy: {accuracy:.4f}")

# Save final model
joblib.dump(final_model, 'final_mlp_model.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')
