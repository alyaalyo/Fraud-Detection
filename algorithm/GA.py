import random
import tensorflow as tf
from tensorflow.keras import layers, models

class LayerGene:
    """
    Representation of a single neural network layer.
    """
    def __init__(self, units, activation, dropout_rate):
        self.units = units          # number of neurons
        self.activation = activation  # activation function
        self.dropout_rate = dropout_rate  # dropout (after the layer)

    def __repr__(self):
        return f"Layer(units={self.units}, activation={self.activation}, dropout={self.dropout_rate})"


class ArchitectureGene:
    """
    Gene encoding a neural network architecture.
    Contains a list of layers (LayerGene) and possibly global parameters.
    """
    def __init__(self, layers):
        self.layers = layers  # list of LayerGene objects

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


def generate_random_individual(
    min_layers=1,
    max_layers=5,
    units_choices=[16, 32, 64, 128, 256],
    activation_choices=['relu', 'tanh', 'sigmoid'],
    dropout_range=(0.0, 0.5),
):
    """
    Generates a random individual (ArchitectureGene) with given constraints.
    
    Parameters:
        min_layers, max_layers: range for the number of layers.
        units_choices: list of allowed layer sizes.
        activation_choices: list of allowed activation functions.
        dropout_range: tuple (min, max) for the dropout rate.
        input_dim: ignored, but can be useful for compatibility checks.
    
    Returns:
        ArchitectureGene
    """
    num_layers = random.randint(min_layers, max_layers)
    layers = []
    for _ in range(num_layers):
        units = random.choice(units_choices)
        activation = random.choice(activation_choices)
        dropout_rate = round(random.uniform(*dropout_range), 2)  # round to two decimals
        layers.append(LayerGene(units, activation, dropout_rate))
    return ArchitectureGene(layers)


def build_mlp_from_gene(gene: ArchitectureGene, input_dim: int) -> models.Sequential:
    """
    Builds a Keras MLP model from an ArchitectureGene.
    
    Parameters:
        gene: ArchitectureGene object containing layer specifications.
        input_dim: number of input features.
    
    Returns:
        A compiled (but not yet trained) Sequential model.
    """
    model = models.Sequential()
    for i, layer_gene in enumerate(gene.layers):
        if i == 0:
            # First layer: specify input_shape
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
    
    # Output layer for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Optionally compile (can be done later during training)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def initialize_population(pop_size: int, input_dim: int, **kwargs) -> list:
    """
    Creates a population of random individuals and builds their MLP models.
    
    Parameters:
        pop_size: number of individuals in the population.
        input_dim: number of input features (used for model building).
        **kwargs: additional arguments passed to generate_random_individual.
    
    Returns:
        List of tuples (individual, model) for each member.
    """
    population = []
    for _ in range(pop_size):
        individual = generate_random_individual(**kwargs)
        model = build_mlp_from_gene(individual, input_dim)
        population.append((individual, model))
    return population


# Example usage
if __name__ == "__main__":
    # Example: input dimension (e.g., after feature extraction)
    INPUT_DIM = 100
    POP_SIZE = 5

    # Initialize population
    population = initialize_population(POP_SIZE, INPUT_DIM)

    # Display each individual and its model summary
    for idx, (ind, model) in enumerate(population):
        print(f"\n--- Individual {idx+1} ---")
        print("Gene:", ind)
        print("Model summary:")
        model.summary()
        print("Total parameters:", model.count_params())