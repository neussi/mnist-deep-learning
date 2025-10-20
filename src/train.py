"""
Script d'entraînement du modèle MNIST avec MLflow tracking
"""

import mlflow
import mlflow.keras
from datetime import datetime
import os

from model import create_model, print_model_summary
from utils import (
    load_mnist_data, 
    preprocess_data, 
    visualize_samples,
    plot_training_history,
    create_confusion_matrix,
    save_model_summary
)


def train_model(epochs=5, batch_size=128, hidden_units=512, dropout_rate=0.2):
    """
    Entraîne le modèle MNIST avec suivi MLflow
    
    Args:
        epochs (int): Nombre d'époques
        batch_size (int): Taille des batches
        hidden_units (int): Neurones dans la couche cachée
        dropout_rate (float): Taux de dropout
    """
    print("\n" + "="*70)
    print(" DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("="*70)
    
    # Démarrer une expérience MLflow
    mlflow.set_experiment("MNIST_Classification")
    
    with mlflow.start_run():
        # Enregistrer les paramètres
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("hidden_units", hidden_units)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("optimizer", "adam")
        
        # Étape 1: Charger les données
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
        
        # Visualiser quelques exemples
        visualize_samples(x_train, y_train)
        mlflow.log_artifact("models/mnist_samples.png")
        
        # Étape 2: Prétraiter les données
        x_train_proc, x_test_proc = preprocess_data(x_train, x_test)
        
        # Étape 3: Créer le modèle
        model = create_model(
            input_shape=784,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate
        )
        
        print_model_summary(model)
        save_model_summary(model)
        mlflow.log_artifact("models/model_summary.txt")
        
        # Étape 4: Entraîner le modèle
        print("\n  Début de l'entraînement...\n")
        
        history = model.fit(
            x_train_proc, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        
        # Étape 5: Évaluer sur le test set
        print("\n Évaluation sur le test set...")
        test_loss, test_accuracy = model.evaluate(x_test_proc, y_test, verbose=0)
        
        print(f"\n Résultats finaux:")
        print(f"   - Test Loss: {test_loss:.4f}")
        print(f"   - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Enregistrer les métriques dans MLflow
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
        
        # Étape 6: Sauvegarder les visualisations
        plot_training_history(history)
        mlflow.log_artifact("models/training_history.png")
        
        create_confusion_matrix(model, x_test_proc, y_test)
        mlflow.log_artifact("models/confusion_matrix.png")
        
        # Étape 7: Sauvegarder le modèle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/mnist_model_{timestamp}.keras"
        model.save(model_path)
        print(f"\n Modèle sauvegardé: {model_path}")
        
        # Enregistrer le modèle dans MLflow
        mlflow.keras.log_model(model, "model")
        
        # Sauvegarder aussi en format .h5 pour l'API
        model.save("models/mnist_model.h5")
        print(f" Modèle sauvegardé pour l'API: models/mnist_model.h5")
        
        print("\n" + "="*70)
        print(" ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("="*70)
        print(f"\n Pour voir les résultats dans MLflow:")
        print("   mlflow ui")
        print("   Puis ouvrir: http://localhost:5000")
        print("\n")
        
        return model, test_accuracy


if __name__ == "__main__":
    # Paramètres d'entraînement
    EPOCHS = 5
    BATCH_SIZE = 128
    HIDDEN_UNITS = 512
    DROPOUT_RATE = 0.2
    
    # Entraîner le modèle
    model, accuracy = train_model(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_units=HIDDEN_UNITS,
        dropout_rate=DROPOUT_RATE
    )