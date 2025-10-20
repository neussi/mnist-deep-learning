# MNIST Deep Learning Engineering Project

Projet complet de Deep Learning avec MLflow, Flask API et Docker.

## Description

Ce projet implémente un système complet de classification de chiffres manuscrits (MNIST) avec :
- Réseau de neurones avec TensorFlow/Keras
- Suivi des expériences avec MLflow
- API REST avec Flask
- Conteneurisation avec Docker
- Pipeline CI/CD

## Installation

### Prérequis
- Python 3.9+
- pip
- Git
- Docker (optionnel)

### Installation rapide
```bash
# Cloner le projet
git clone git@github.com:neussi/mnist-deep-learning.git
cd mnist-deep-learning

# Exécuter le script d'installation
chmod +x setup.sh
./setup.sh
```

### Installation manuelle
```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Structure du Projet
```
mnist-deep-learning/
├── src/           # Code source
├── api/           # API Flask
├── models/        # Modèles sauvegardés
├── tests/         # Tests
└── README.md      # Ce fichier
```

## Utilisation

### 1. Entraîner le modèle
```bash
python src/train.py
```

### 2. Lancer MLflow UI
```bash
mlflow ui
```
Ouvrir http://localhost:5000

### 3. Lancer l'API
```bash
python api/app.py
```

### 4. Tester l'API
```bash
curl http://localhost:5000/health
```

## Tests
```bash
pytest tests/
```

## Docker
```bash
# Construire l'image
docker build -t mnist-api .

# Lancer le conteneur
docker run -p 5000:5000 mnist-api
```

##  Résultats

- Accuracy: 97.8%
- Test Loss: 0.072
- Latence API: ~45ms

##  Auteur

NEUSSI PATRICE - ENSPY Génie Informatique (AIA4)

## Licence

MIT License