import random
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Dict, Any
import hashlib
import json

class LayerGene:
    """
    Representation of a single neural network layer.
    """
    def __init__(self, units, activation, dropout_rate):
        self.units = units          # number of neurons
        self.activation = activation  # activation function
        self.dropout_rate = dropout_rate  # dropout (after the layer)
        
    def __eq__(self, other):
        if not isinstance(other, LayerGene):
            return False
        return (self.units == other.units and 
                self.activation == other.activation and 
                self.dropout_rate == other.dropout_rate)
    
    def __hash__(self):
        return hash((self.units, self.activation, self.dropout_rate))

    def __repr__(self):
        return f"Layer(units={self.units}, activation={self.activation}, dropout={self.dropout_rate})"


class ArchitectureGene:
    """
    Gene encoding a neural network architecture.
    Contains a list of layers (LayerGene) and possibly global parameters.
    """
    def __init__(self, layers):
        self.layers = layers  # list of LayerGene objects
        
    def __eq__(self, other):
        if not isinstance(other, ArchitectureGene):
            return False
        return self.layers == other.layers
    
    def __hash__(self):
        return hash(tuple(self.layers))

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        return f"ArchitectureGene(layers={self.layers})"

    def to_dict(self):
        """Represent the gene as a dictionary (for serialization)."""
        return {
            'layers': [
                {'units': l.units, 'activation': l.activation, 'dropout_rate': l.dropout_rate}
                for l in self.layers
            ]
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create ArchitectureGene from dictionary."""
        layers = [LayerGene(l['units'], l['activation'], l['dropout_rate']) 
                  for l in data['layers']]
        return cls(layers)


class ModelCache:
    """Cache for built models to avoid redundant construction."""
    def __init__(self, maxsize=128):
        self.cache = {}
        self.maxsize = maxsize
        self.access_count = {}
    
    def get_key(self, gene: ArchitectureGene, input_dim: int) -> str:
        """Generate unique key for gene and input dimension."""
        gene_dict = gene.to_dict()
        key_data = {
            'gene': gene_dict,
            'input_dim': input_dim
        }
        # Use deterministic serialization
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, gene: ArchitectureGene, input_dim: int):
        """Retrieve model from cache if exists."""
        key = self.get_key(gene, input_dim)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, gene: ArchitectureGene, input_dim: int, model):
        """Store model in cache with LRU eviction."""
        key = self.get_key(gene, input_dim)
        
        if len(self.cache) >= self.maxsize:
            # Evict least recently used item
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = model
        self.access_count[key] = 1
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()


# Global cache instance
_model_cache = ModelCache()


def generate_random_individual(
    min_layers=1,
    max_layers=5,
    units_choices=[16, 32, 64, 128, 256],
    activation_choices=['relu', 'tanh', 'sigmoid'],
    dropout_range=(0.0, 0.5),
):
    """
    Generates a random individual (ArchitectureGene) with given constraints.
    Vectorized generation for multiple individuals.
    """
    # Pre-compute random choices for vectorization
    num_layers = random.randint(min_layers, max_layers)
    layers = []
    
    # Vectorized selection of parameters
    units_indices = np.random.randint(0, len(units_choices), num_layers)
    activation_indices = np.random.randint(0, len(activation_choices), num_layers)
    dropout_rates = np.random.uniform(*dropout_range, num_layers)
    
    for i in range(num_layers):
        units = units_choices[units_indices[i]]
        activation = activation_choices[activation_indices[i]]
        dropout_rate = round(float(dropout_rates[i]), 2)
        layers.append(LayerGene(units, activation, dropout_rate))
    return ArchitectureGene(layers)


def generate_population_vectorized(pop_size: int, **kwargs) -> List[ArchitectureGene]:
    """
    Vectorized generation of multiple individuals.
    """
    min_layers = kwargs.get('min_layers', 1)
    max_layers = kwargs.get('max_layers', 5)
    units_choices = kwargs.get('units_choices', [16, 32, 64, 128, 256])
    activation_choices = kwargs.get('activation_choices', ['relu', 'tanh', 'sigmoid'])
    dropout_range = kwargs.get('dropout_range', (0.0, 0.5))
    
    # Generate all layer counts at once
    num_layers_list = np.random.randint(min_layers, max_layers + 1, pop_size)
    
    individuals = []
    for num_layers in num_layers_list:
        # Vectorized parameter generation per individual
        units_indices = np.random.randint(0, len(units_choices), num_layers)
        activation_indices = np.random.randint(0, len(activation_choices), num_layers)
        dropout_rates = np.random.uniform(*dropout_range, num_layers)
        
        layers = []
        for i in range(num_layers):
            units = units_choices[units_indices[i]]
            activation = activation_choices[activation_indices[i]]
            dropout_rate = round(float(dropout_rates[i]), 2)
            layers.append(LayerGene(units, activation, dropout_rate))
        
        individuals.append(ArchitectureGene(layers))
    
    return individuals


@lru_cache(maxsize=128)
def _build_mlp_from_gene_cached(gene_dict_str: str, input_dim: int) -> models.Sequential:
    """
    Cached version of model building using serialized gene.
    """
    gene_dict = json.loads(gene_dict_str)
    gene = ArchitectureGene.from_dict(gene_dict)
    
    model = models.Sequential()
    for i, layer_gene in enumerate(gene.layers):
        if i == 0:
            model.add(layers.Dense(
                layer_gene.units,
                activation=layer_gene.activation,
                input_shape=(input_dim,)
            ))
        else:
            model.add(layers.Dense(
                layer_gene.units,
                activation=layer_gene.activation
            ))
        if layer_gene.dropout_rate > 0:
            model.add(layers.Dropout(layer_gene.dropout_rate))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def build_mlp_from_gene(gene: ArchitectureGene, input_dim: int) -> models.Sequential:
    """
    Builds a Keras MLP model from an ArchitectureGene with caching.
    """
    gene_dict_str = json.dumps(gene.to_dict(), sort_keys=True)
    return _build_mlp_from_gene_cached(gene_dict_str, input_dim)


def initialize_population(pop_size: int, input_dim: int, **kwargs) -> list:
    """
    Creates a population of random individuals and builds their MLP models.
    Optimized with vectorized generation and caching.
    """
    # Vectorized individual generation
    individuals = generate_population_vectorized(pop_size, **kwargs)
    
    # Build models with caching
    population = []
    for individual in individuals:
        model = build_mlp_from_gene(individual, input_dim)
        population.append((individual, model))
    return population


def mutate_individual_vectorized(individuals: List[ArchitectureGene], 
                                 mutation_rate: float = 0.3,
                                 units_choices=[16, 32, 64, 128, 256],
                                 activation_choices=['relu', 'tanh', 'sigmoid'],
                                 dropout_range=(0.0, 0.5)) -> List[ArchitectureGene]:
    """
    Vectorized mutation of multiple individuals.
    """
    mutated_individuals = []
    
    for individual in individuals:
        mutated_layers = []
        for layer in individual.layers:
            if random.random() < mutation_rate:
                # Pre-select mutation types for better performance
                mutation_type = random.choice(['units', 'activation', 'dropout'])
                
                units = layer.units
                activation = layer.activation
                dropout_rate = layer.dropout_rate
                
                if mutation_type == 'units':
                    units = random.choice(units_choices)
                elif mutation_type == 'activation':
                    activation = random.choice(activation_choices)
                else:  # dropout
                    dropout_rate = round(random.uniform(*dropout_range), 2)
                    
                mutated_layers.append(LayerGene(units, activation, dropout_rate))
            else:
                mutated_layers.append(layer)
        
        mutated_individuals.append(ArchitectureGene(mutated_layers))
    
    return mutated_individuals


def mutate_individual(individual: ArchitectureGene, mutation_rate: float = 0.3, 
                      units_choices=[16, 32, 64, 128, 256],
                      activation_choices=['relu', 'tanh', 'sigmoid'],
                      dropout_range=(0.0, 0.5)) -> ArchitectureGene:
    """
    Mutates an individual by randomly changing layer parameters.
    Each layer has a probability mutation_rate to be mutated.
    """
    mutated_layers = []
    for layer in individual.layers:
        if random.random() < mutation_rate:
            mutation_type = random.choice(['units', 'activation', 'dropout'])
            
            units = layer.units
            activation = layer.activation
            dropout_rate = layer.dropout_rate
            
            if mutation_type == 'units':
                units = random.choice(units_choices)
            elif mutation_type == 'activation':
                activation = random.choice(activation_choices)
            else:  # dropout
                dropout_rate = round(random.uniform(*dropout_range), 2)
                
            mutated_layers.append(LayerGene(units, activation, dropout_rate))
        else:
            mutated_layers.append(layer)
    
    return ArchitectureGene(mutated_layers)

def roulette_selection(population_with_fitness, num_parents: int):
    """
    Selects parents using roulette wheel selection based on fitness scores.
    Vectorized version for efficiency.
    """
    fitness_values = np.array([fitness for _, _, fitness in population_with_fitness])
    
    # Handle negative fitness values by shifting to positive
    min_fitness = np.min(fitness_values)
    if min_fitness < 0:
        fitness_values = fitness_values - min_fitness + 1e-10
    
    # Calculate total fitness and selection probabilities
    total_fitness = np.sum(fitness_values)
    if total_fitness > 0:
        selection_probs = fitness_values / total_fitness
    else:
        # If all fitness are zero or negative, use uniform distribution
        selection_probs = np.ones(len(fitness_values)) / len(fitness_values)
    
    # Vectorized selection
    selected_indices = np.random.choice(
        len(population_with_fitness), 
        size=num_parents, 
        p=selection_probs, 
        replace=False
    )
    
    return [population_with_fitness[i][0] for i in selected_indices]


def single_point_crossover(parent1: ArchitectureGene, parent2: ArchitectureGene) -> tuple:
    """
    Performs single-point crossover between two parent architectures.
    Returns two child individuals.
    """
    min_len = min(len(parent1), len(parent2))
    if min_len < 2:
        return parent1, parent2
    
    crossover_point = random.randint(1, min_len - 1)
    
    child1_layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
    child2_layers = parent2.layers[:crossover_point] + parent1.layers[crossover_point:]
    
    return ArchitectureGene(child1_layers), ArchitectureGene(child2_layers)


def elitism_selection(population_with_fitness, num_elites: int) -> List:
    """
    Select the best individuals (elites) based on fitness.
    
    Parameters:
        population_with_fitness: list of tuples [(individual, model, fitness), ...]
        num_elites: number of elites to select
    
    Returns:
        List of elite individuals (without models)
    """
    # Sort by fitness descending
    sorted_population = sorted(population_with_fitness, key=lambda x: x[2], reverse=True)
    
    # Select top elites
    elites = [individual for individual, _, _ in sorted_population[:num_elites]]
    
    return elites


def evolutionary_step(population_with_fitness, mutation_rate=0.3, crossover_prob=0.7, 
                      num_elites=2, num_parents=None):
    """
    Perform one evolutionary step with elitism.
    
    Parameters:
        population_with_fitness: list of [(individual, model, fitness)]
        mutation_rate: probability of mutation per layer
        crossover_prob: probability of crossover
        num_elites: number of elite individuals to preserve
        num_parents: number of parents to select (defaults to pop_size - num_elites)
    
    Returns:
        New population list of (individual, model) pairs
    """
    pop_size = len(population_with_fitness)
    
    if num_parents is None:
        num_parents = pop_size - num_elites
    
    # Select elites
    elites = elitism_selection(population_with_fitness, num_elites)
    elite_models = [model for _, model, _ in population_with_fitness[:num_elites]]
    
    # Select parents using roulette wheel
    parents = roulette_selection(population_with_fitness, num_parents)
    
    # Create offspring through crossover and mutation
    offspring = []
    
    # Process parents in pairs
    for i in range(0, len(parents) - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        
        if random.random() < crossover_prob:
            child1, child2 = single_point_crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2
        
        # Mutate children
        child1 = mutate_individual(child1, mutation_rate)
        child2 = mutate_individual(child2, mutation_rate)
        
        offspring.append(child1)
        offspring.append(child2)
    
    # If odd number of parents, add last parent as is
    if len(parents) % 2 == 1:
        offspring.append(mutate_individual(parents[-1], mutation_rate))
    
    # Build models for offspring
    input_dim = population_with_fitness[0][1].input_shape[1]
    new_population = [(individual, build_mlp_from_gene(individual, input_dim)) 
                      for individual in offspring]
    
    # Add elites
    new_population.extend(zip(elites, elite_models))
    
    return new_population


def clear_model_cache():
    """Clear the model building cache."""
    _model_cache.clear()
    _build_mlp_from_gene_cached.cache_clear()


# Example usage
if __name__ == "__main__":
    # Example: input dimension (e.g., after feature extraction)
    INPUT_DIM = 100
    POP_SIZE = 10
    NUM_GENERATIONS = 5
    
    # Clear cache at start
    clear_model_cache()
    
    # Initialize population
    print("Initializing population...")
    population = initialize_population(POP_SIZE, INPUT_DIM)
    
    # Simulate fitness evaluation (replace with actual evaluation)
    population_with_fitness = [(ind, model, random.random()) for ind, model in population]
    
    # Run evolutionary process with elitism
    for generation in range(NUM_GENERATIONS):
        print(f"\n--- Generation {generation + 1} ---")
        
        # Perform evolutionary step
        population = evolutionary_step(
            population_with_fitness,
            mutation_rate=0.3,
            crossover_prob=0.7,
            num_elites=2
        )
        
        # Re-evaluate fitness (simulated)
        population_with_fitness = [(ind, model, random.random()) for ind, model in population]
        
        # Display best fitness
        best_fitness = max(population_with_fitness, key=lambda x: x[2])[2]
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Population size: {len(population)}")
    
    # Display final population
    for idx, (ind, model) in enumerate(population[:3]):  # Show only first 3
        print(f"\n--- Final Individual {idx+1} ---")
        print("Gene:", ind)
        print("Total parameters:", model.count_params())
    
    # Show cache statistics
    cache_info = _build_mlp_from_gene_cached.cache_info()
    print(f"\nCache statistics: {cache_info}")