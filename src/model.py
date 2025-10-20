"""
Définition du modèle de réseau de neurones pour MNIST
"""

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_model(input_shape=784, hidden_units=512, dropout_rate=0.2, num_classes=10):
    """
    Crée un réseau de neurones fully-connected pour MNIST
    
    Architecture:
        - Input Layer: 784 neurones (28x28 pixels)
        - Hidden Layer 1: 512 neurones + ReLU + Dropout
        - Output Layer: 10 neurones (classes 0-9) + Softmax
    
    Args:
        input_shape (int): Taille de l'entrée (784 pour MNIST)
        hidden_units (int): Nombre de neurones dans la couche cachée
        dropout_rate (float): Taux de dropout (0.0 à 1.0)
        num_classes (int): Nombre de classes (10 pour MNIST)
        
    Returns:
        model: Modèle Keras compilé
    """
    print("\n  Construction du modèle...")
    
    model = keras.Sequential([
        # Couche d'entrée
        layers.Input(shape=(input_shape,), name='input_layer'),
        
        # Couche cachée 1
        layers.Dense(
            hidden_units, 
            activation='relu',
            kernel_initializer='he_normal',
            name='hidden_layer_1'
        ),
        
        # Dropout pour la régularisation
        layers.Dropout(dropout_rate, name='dropout_layer'),
        
        # Couche de sortie
        layers.Dense(
            num_classes, 
            activation='softmax',
            name='output_layer'
        )
    ])
    
    # Compilation du modèle
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(" Modèle créé et compilé")
    print(f"   - Couche cachée: {hidden_units} neurones")
    print(f"   - Dropout: {dropout_rate}")
    print(f"   - Optimiseur: Adam")
    print(f"   - Fonction de perte: sparse_categorical_crossentropy")
    
    return model


def get_model_info(model):
    """
    Retourne les informations du modèle
    
    Args:
        model: Modèle Keras
        
    Returns:
        dict: Dictionnaire avec les infos du modèle
    """
    total_params = model.count_params()
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': sum([np.prod(v.shape) for v in model.trainable_weights]),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    
    return info


def print_model_summary(model):
    """
    Affiche un résumé détaillé du modèle
    
    Args:
        model: Modèle Keras
    """
    print("\n" + "="*60)
    print(" RÉSUMÉ DU MODÈLE")
    print("="*60)
    
    model.summary()
    
    info = get_model_info(model)
    
    print("\n" + "="*60)
    print(" STATISTIQUES")
    print("="*60)
    print(f"Nombre total de paramètres: {info['total_parameters']:,}")
    print(f"Paramètres entraînables: {info['trainable_parameters']:,}")
    print(f"Nombre de couches: {info['layers']}")
    print(f"Forme d'entrée: {info['input_shape']}")
    print(f"Forme de sortie: {info['output_shape']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test du modèle
    print(" Test de création du modèle\n")
    
    # Créer le modèle
    model = create_model()
    
    # Afficher le résumé
    print_model_summary(model)
    
    # Test avec des données aléatoires
    print(" Test de prédiction avec données aléatoires...")
    random_input = np.random.rand(1, 784)
    prediction = model.predict(random_input, verbose=0)
    
    print(f"   Prédiction réussie!")
    print(f"   Shape de sortie: {prediction.shape}")
    print(f"   Somme des probabilités: {prediction.sum():.4f}")
    print(f"   Classe prédite: {np.argmax(prediction)}")