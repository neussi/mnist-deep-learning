"""
Fonctions utilitaires pour le projet MNIST
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os


def load_mnist_data():
    """
    Charge et prépare le dataset MNIST
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    print(" Chargement du dataset MNIST...")
    
    # Charger MNIST depuis Keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f" Données chargées:")
    print(f"   - Training set: {x_train.shape}")
    print(f"   - Test set: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, x_test):
    """
    Prétraite les données MNIST
    
    Args:
        x_train: Images d'entraînement
        x_test: Images de test
        
    Returns:
        tuple: (x_train_processed, x_test_processed)
    """
    print("\n Prétraitement des données...")
    
    # Aplatir les images 28x28 en vecteurs de 784
    x_train_flat = x_train.reshape(x_train.shape[0], 784)
    x_test_flat = x_test.reshape(x_test.shape[0], 784)
    
    # Normalisation : pixels de [0, 255] à [0, 1]
    x_train_norm = x_train_flat.astype('float32') / 255.0
    x_test_norm = x_test_flat.astype('float32') / 255.0
    
    print(f" Prétraitement terminé:")
    print(f"   - Shape après reshape: {x_train_norm.shape}")
    print(f"   - Valeurs normalisées: [{x_train_norm.min():.2f}, {x_train_norm.max():.2f}]")
    
    return x_train_norm, x_test_norm


def visualize_samples(x_data, y_data, n_samples=10):
    """
    Affiche quelques exemples du dataset
    
    Args:
        x_data: Images (format original 28x28)
        y_data: Labels
        n_samples: Nombre d'exemples à afficher
    """
    plt.figure(figsize=(15, 3))
    
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(x_data[i], cmap='gray')
        plt.title(f'Label: {y_data[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/mnist_samples.png', dpi=150, bbox_inches='tight')
    print(f"\n Exemples sauvegardés dans 'models/mnist_samples.png'")
    plt.close()


def plot_training_history(history, save_path='models/training_history.png'):
    """
    Affiche les courbes d'entraînement
    
    Args:
        history: Historique Keras
        save_path: Chemin de sauvegarde
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy au cours de l\'entraînement')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss au cours de l\'entraînement')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Courbes d'entraînement sauvegardées dans '{save_path}'")
    plt.close()


def create_confusion_matrix(model, x_test, y_test):
    """
    Crée une matrice de confusion
    
    Args:
        model: Modèle entraîné
        x_test: Données de test
        y_test: Labels de test
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # Prédictions
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Visualisation
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"\n Matrice de confusion sauvegardée dans 'models/confusion_matrix.png'")
    plt.close()
    
    # Rapport de classification
    print("\n Rapport de Classification:")
    print(classification_report(y_test, y_pred_classes, 
                                target_names=[str(i) for i in range(10)]))


def save_model_summary(model, filepath='models/model_summary.txt'):
    """
    Sauvegarde le résumé du modèle dans un fichier
    
    Args:
        model: Modèle Keras
        filepath: Chemin de sauvegarde
    """
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"\n Résumé du modèle sauvegardé dans '{filepath}'")


if __name__ == "__main__":
    # Test des fonctions
    print("Test des fonctions utilitaires\n")
    
    # Charger les données
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Visualiser quelques exemples
    visualize_samples(x_train, y_train)
    
    # Prétraiter
    x_train_proc, x_test_proc = preprocess_data(x_train, x_test)
    
    print("\n Tous les tests sont passés!")