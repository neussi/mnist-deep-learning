"""
API REST Flask pour la prédiction de chiffres MNIST
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Créer l'application Flask
app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite à 16MB
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/mnist_model.h5')

# Variable globale pour le modèle
model = None


def load_model():
    """
    Charge le modèle TensorFlow/Keras
    """
    global model
    
    try:
        logger.info(f" Chargement du modèle depuis {MODEL_PATH}...")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f" Modèle non trouvé: {MODEL_PATH}")
            logger.info(" Veuillez d'abord entraîner le modèle avec: python src/train.py")
            return False
        
        model = keras.models.load_model(MODEL_PATH)
        logger.info(" Modèle chargé avec succès!")
        
        # Faire une prédiction dummy pour "warm up" le modèle
        dummy_input = np.zeros((1, 784))
        _ = model.predict(dummy_input, verbose=0)
        logger.info(" Warm-up du modèle terminé")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return False


def preprocess_image(image_data):
    """
    Prétraite l'image pour la prédiction
    
    Args:
        image_data: Liste ou array de 784 valeurs (0-255)
        
    Returns:
        np.array: Image prétraitée
    """
    try:
        # Convertir en numpy array
        image = np.array(image_data, dtype=np.float32)
        
        # Vérifier la taille
        if image.shape[0] != 784:
            raise ValueError(f"L'image doit contenir 784 pixels, reçu {image.shape[0]}")
        
        # Normaliser [0, 255] -> [0, 1]
        image = image / 255.0
        
        # Ajouter dimension batch
        image = image.reshape(1, 784)
        
        return image
        
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement: {str(e)}")
        raise


@app.route('/', methods=['GET'])
def home():
    """
    Page d'accueil de l'API
    """
    return jsonify({
        'service': 'MNIST Digit Recognition API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'GET /': 'Informations sur l\'API',
            'GET /health': 'Vérification de santé',
            'POST /predict': 'Prédiction d\'un chiffre',
            'GET /model/info': 'Informations sur le modèle'
        },
        'documentation': 'https://github.com/neussi/mnist-deep-learning',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de santé pour vérifier que l'API fonctionne
    """
    if model is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Modèle non chargé',
            'timestamp': datetime.now().isoformat()
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'message': 'API opérationnelle',
        'model_loaded': True,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Retourne les informations sur le modèle
    """
    if model is None:
        return jsonify({
            'error': 'Modèle non chargé'
        }), 503
    
    try:
        return jsonify({
            'model_type': 'Sequential Neural Network',
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_parameters': int(model.count_params()),
            'layers': len(model.layers),
            'framework': 'TensorFlow/Keras',
            'task': 'Digit Classification (0-9)',
            'dataset': 'MNIST'
        })
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction
    
    Body JSON attendu:
    {
        "image": [0, 0, ..., 255]  // 784 valeurs entre 0 et 255
    }
    
    Retourne:
    {
        "prediction": 7,
        "confidence": 0.9876,
        "probabilities": [0.001, 0.002, ..., 0.987],
        "processing_time_ms": 45.2
    }
    """
    if model is None:
        return jsonify({
            'error': 'Modèle non chargé',
            'message': 'Le serveur n\'est pas prêt'
        }), 503
    
    try:
        start_time = datetime.now()
        
        # Vérifier le Content-Type
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type doit être application/json'
            }), 400
        
        # Récupérer les données
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'error': 'Clé "image" manquante dans le JSON',
                'example': {
                    'image': [0] * 784
                }
            }), 400
        
        image_data = data['image']
        
        # Vérifier le type
        if not isinstance(image_data, (list, np.ndarray)):
            return jsonify({
                'error': 'L\'image doit être une liste ou un array'
            }), 400
        
        # Prétraiter l'image
        processed_image = preprocess_image(image_data)
        
        # Faire la prédiction
        predictions = model.predict(processed_image, verbose=0)
        probabilities = predictions[0].tolist()
        
        # Obtenir la classe prédite
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(probabilities[predicted_class])
        
        # Calculer le temps de traitement
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f" Prédiction: {predicted_class} (confiance: {confidence:.4f})")
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'processing_time_ms': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except ValueError as e:
        logger.warning(f" Erreur de validation: {str(e)}")
        return jsonify({
            'error': 'Données invalides',
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f" Erreur lors de la prédiction: {str(e)}")
        return jsonify({
            'error': 'Erreur interne du serveur',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Gestion des erreurs 404"""
    return jsonify({
        'error': 'Endpoint non trouvé',
        'message': 'Veuillez consulter la documentation',
        'available_endpoints': ['/', '/health', '/predict', '/model/info']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs 500"""
    logger.error(f"Erreur 500: {str(error)}")
    return jsonify({
        'error': 'Erreur interne du serveur',
        'message': 'Une erreur inattendue s\'est produite'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" DÉMARRAGE DE L'API MNIST")
    print("="*70)
    
    # Charger le modèle
    if load_model():
        print("\n API prête à recevoir des requêtes!")
        print("="*70)
        print("\n Endpoints disponibles:")
        print("   - GET  http://localhost:5000/")
        print("   - GET  http://localhost:5000/health")
        print("   - POST http://localhost:5000/predict")
        print("   - GET  http://localhost:5000/model/info")
        print("\n Exemple de requête:")
        print("   curl -X POST http://localhost:5000/predict \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"image\": [0, 0, ..., 255]}'")
        print("\n" + "="*70 + "\n")
        
        # Lancer le serveur
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False  # Mettre True pour le développement
        )
    else:
        print("\n Impossible de démarrer l'API sans modèle")
        print("   Veuillez d'abord entraîner le modèle:")
        print("   python src/train.py")
        print("\n")