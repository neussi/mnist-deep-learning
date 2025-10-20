"""
Script de tests pour l'API Flask MNIST
"""

import requests
import numpy as np
from tensorflow import keras
import json
import time


# Configuration
API_URL = "http://localhost:5000"
HEADERS = {'Content-Type': 'application/json'}


def print_separator(title=""):
    """Affiche un séparateur"""
    print("\n" + "="*70)
    if title:
        print(f" {title}")
        print("="*70)


def test_home():
    """Test de l'endpoint GET /"""
    print_separator("TEST 1: Endpoint Home (GET /)")
    
    try:
        response = requests.get(f"{API_URL}/")
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Status code devrait être 200"
        assert 'service' in response.json(), "Clé 'service' manquante"
        
        print(" Test réussi!")
        return True
        
    except Exception as e:
        print(f" Test échoué: {str(e)}")
        return False


def test_health():
    """Test de l'endpoint GET /health"""
    print_separator("TEST 2: Endpoint Health (GET /health)")
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Status code devrait être 200"
        assert response.json()['status'] == 'healthy', "Status devrait être 'healthy'"
        
        print(" Test réussi!")
        return True
        
    except Exception as e:
        print(f" Test échoué: {str(e)}")
        return False


def test_model_info():
    """Test de l'endpoint GET /model/info"""
    print_separator("TEST 3: Endpoint Model Info (GET /model/info)")
    
    try:
        response = requests.get(f"{API_URL}/model/info")
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Status code devrait être 200"
        assert 'total_parameters' in response.json(), "Clé 'total_parameters' manquante"
        
        print(" Test réussi!")
        return True
        
    except Exception as e:
        print(f" Test échoué: {str(e)}")
        return False


def test_predict_with_real_data():
    """Test de prédiction avec de vraies données MNIST"""
    print_separator("TEST 4: Prédiction avec données réelles")
    
    try:
        # Charger une image MNIST réelle
        print(" Chargement d'une image MNIST...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Prendre la première image de test
        test_image = x_test[0]
        true_label = y_test[0]
        
        print(f"Label réel: {true_label}")
        
        # Prétraiter : aplatir en vecteur de 784
        image_flat = test_image.reshape(784).tolist()
        
        # Préparer le payload
        payload = {
            "image": image_flat
        }
        
        # Faire la requête
        print(" Envoi de la requête...")
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            headers=HEADERS,
            json=payload
        )
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Temps de réponse: {(end_time - start_time)*1000:.2f} ms")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Status code devrait être 200"
        
        result = response.json()
        predicted_label = result['prediction']
        confidence = result['confidence']
        
        print(f"\n Résultats:")
        print(f"   - Label réel: {true_label}")
        print(f"   - Label prédit: {predicted_label}")
        print(f"   - Confiance: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   - Correct: {' OUI' if predicted_label == true_label else '❌ NON'}")
        
        print("\n Test réussi!")
        return True
        
    except Exception as e:
        print(f" Test échoué: {str(e)}")
        return False


def test_predict_with_zeros():
    """Test de prédiction avec une image de zéros"""
    print_separator("TEST 5: Prédiction avec image vide (zéros)")
    
    try:
        payload = {
            "image": [0] * 784
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            headers=HEADERS,
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Status code devrait être 200"
        assert 'prediction' in response.json(), "Clé 'prediction' manquante"
        
        print(" Test réussi!")
        return True
        
    except Exception as e:
        print(f" Test échoué: {str(e)}")
        return False


def test_predict_with_invalid_data():
    """Test avec des données invalides"""
    print_separator("TEST 6: Validation - Données invalides")
    
    test_cases = [
        {
            "name": "Image trop courte",
            "payload": {"image": [0] * 100},
            "expected_status": 400
        },
        {
            "name": "Image trop longue",
            "payload": {"image": [0] * 1000},
            "expected_status": 400
        },
        {
            "name": "Clé manquante",
            "payload": {"data": [0] * 784},
            "expected_status": 400
        },
        {
            "name": "Type invalide",
            "payload": {"image": "invalid"},
            "expected_status": 400
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n  🔍 Test: {test_case['name']}")
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                headers=HEADERS,
                json=test_case['payload']
            )
            
            print(f"     Status: {response.status_code}")
            
            if response.status_code == test_case['expected_status']:
                print(f"      Comportement correct")
            else:
                print(f"      Status attendu: {test_case['expected_status']}")
                all_passed = False
                
        except Exception as e:
            print(f"      Erreur: {str(e)}")
            all_passed = False
    
    if all_passed:
        print("\n Tous les tests de validation réussis!")
    else:
        print("\n Certains tests ont échoué")
    
    return all_passed


def test_multiple_predictions():
    """Test de plusieurs prédictions consécutives"""
    print_separator("TEST 7: Prédictions multiples (performance)")
    
    try:
        # Charger des images
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        n_predictions = 10
        latencies = []
        correct = 0
        
        print(f" Envoi de {n_predictions} requêtes...")
        
        for i in range(n_predictions):
            test_image = x_test[i]
            true_label = y_test[i]
            image_flat = test_image.reshape(784).tolist()
            
            payload = {"image": image_flat}
            
            start_time = time.time()
            response = requests.post(
                f"{API_URL}/predict",
                headers=HEADERS,
                json=payload
            )
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            if response.status_code == 200:
                predicted = response.json()['prediction']
                if predicted == true_label:
                    correct += 1
            
            print(f"   [{i+1}/{n_predictions}] Latence: {latency:.2f}ms - "
                  f"Prédit: {predicted} - Réel: {true_label} "
                  f"{'' if predicted == true_label else ''}")
        
        # Statistiques
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        accuracy = (correct / n_predictions) * 100
        
        print(f"\n   Statistiques:")
        print(f"   - Latence moyenne: {avg_latency:.2f}ms")
        print(f"   - Latence P95: {p95_latency:.2f}ms")
        print(f"   - Accuracy: {accuracy:.1f}% ({correct}/{n_predictions})")
        print(f"   - Throughput: ~{1000/avg_latency:.1f} req/s")
        
        print("\n Test réussi!")
        return True
        
    except Exception as e:
        print(f" Test échoué: {str(e)}")
        return False


def run_all_tests():
    """Exécute tous les tests"""
    print("\n" + "="*70)
    print(" SUITE DE TESTS COMPLÈTE - API MNIST")
    print("="*70)
    
    tests = [
        ("Home Endpoint", test_home),
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Prédiction Réelle", test_predict_with_real_data),
        ("Prédiction Zéros", test_predict_with_zeros),
        ("Validation Erreurs", test_predict_with_invalid_data),
        ("Performance", test_multiple_predictions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n Erreur lors de l'exécution du test '{test_name}': {str(e)}")
            results.append((test_name, False))
    
    # Résumé
    print("\n" + "="*70)
    print(" RÉSUMÉ DES TESTS")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = " PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{status:12} - {test_name}")
    
    print("="*70)
    print(f"Résultat: {passed}/{total} tests réussis ({passed/total*100:.1f}%)")
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    print("\n Vérification que l'API est accessible...")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print(" API accessible!\n")
            run_all_tests()
        else:
            print(" API accessible mais non healthy")
            print(" Vérifiez que le modèle est bien chargé\n")
    except requests.exceptions.ConnectionError:
        print(" Impossible de se connecter à l'API")
        print(f" Assurez-vous que l'API est lancée sur {API_URL}")
        print("   Commande: python api/app.py\n")
    except Exception as e:
        print(f" Erreur: {str(e)}\n")