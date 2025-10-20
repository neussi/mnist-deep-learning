"""
Tests unitaires pour le modèle MNIST
"""

import pytest
import numpy as np
import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import create_model, get_model_info
from src.utils import load_mnist_data, preprocess_data


class TestModel:
    """Tests pour le modèle"""
    
    def test_model_creation(self):
        """Test de création du modèle"""
        model = create_model()
        
        assert model is not None
        assert len(model.layers) == 4  # Input, Dense, Dropout, Output
        
    def test_model_input_shape(self):
        """Test de la forme d'entrée"""
        model = create_model(input_shape=784)
        
        assert model.input_shape == (None, 784)
        
    def test_model_output_shape(self):
        """Test de la forme de sortie"""
        model = create_model(num_classes=10)
        
        assert model.output_shape == (None, 10)
        
    def test_model_parameters(self):
        """Test du nombre de paramètres"""
        model = create_model(hidden_units=512)
        info = get_model_info(model)
        
        # 784*512 + 512 (première couche) + 512*10 + 10 (sortie)
        expected_params = 784*512 + 512 + 512*10 + 10
        
        assert info['total_parameters'] == expected_params
        
    def test_model_prediction_shape(self):
        """Test de la forme de prédiction"""
        model = create_model()
        
        # Créer une entrée aléatoire
        test_input = np.random.rand(1, 784)
        
        # Prédire
        prediction = model.predict(test_input, verbose=0)
        
        assert prediction.shape == (1, 10)
        
    def test_model_probability_sum(self):
        """Test que les probabilités somment à 1"""
        model = create_model()
        
        test_input = np.random.rand(5, 784)
        predictions = model.predict(test_input, verbose=0)
        
        # Vérifier que chaque prédiction somme à ~1
        for pred in predictions:
            assert np.isclose(pred.sum(), 1.0, atol=1e-6)


class TestDataProcessing:
    """Tests pour le traitement des données"""
    
    def test_data_loading(self):
        """Test de chargement des données"""
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
        
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        assert len(y_train) == 60000
        assert len(y_test) == 10000
        
    def test_data_preprocessing(self):
        """Test du prétraitement"""
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
        
        x_train_proc, x_test_proc = preprocess_data(x_train, x_test)
        
        # Vérifier la forme
        assert x_train_proc.shape == (60000, 784)
        assert x_test_proc.shape == (10000, 784)
        
        # Vérifier la normalisation
        assert x_train_proc.min() >= 0.0
        assert x_train_proc.max() <= 1.0
        assert x_test_proc.min() >= 0.0
        assert x_test_proc.max() <= 1.0


class TestIntegration:
    """Tests d'intégration"""
    
    def test_full_pipeline(self):
        """Test du pipeline complet"""
        # Charger les données
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
        
        # Prétraiter
        x_train_proc, x_test_proc = preprocess_data(x_train, x_test)
        
        # Créer le modèle
        model = create_model()
        
        # Entraîner sur un petit échantillon
        model.fit(
            x_train_proc[:1000], 
            y_train[:1000],
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        # Évaluer
        loss, accuracy = model.evaluate(
            x_test_proc[:100], 
            y_test[:100],
            verbose=0
        )
        
        # L'accuracy devrait être > 10% (random guess)
        assert accuracy > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])