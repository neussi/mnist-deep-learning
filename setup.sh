#!/bin/bash

echo " Installation du projet MNIST Deep Learning"
echo "=============================================="

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vérifier Python
echo -e "\n${YELLOW} Vérification de Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED} Python 3 n'est pas installé${NC}"
    exit 1
fi
echo -e "${GREEN} Python $(python3 --version) détecté${NC}"

# Créer l'environnement virtuel
echo -e "\n${YELLOW}  Création de l'environnement virtuel...${NC}"
python3 -m venv venv

# Activer l'environnement
echo -e "\n${YELLOW} Activation de l'environnement...${NC}"
source venv/bin/activate

# Mettre à jour pip
echo -e "\n${YELLOW} Mise à jour de pip...${NC}"
pip install --upgrade pip

# Installer les dépendances
echo -e "\n${YELLOW} Installation des dépendances...${NC}"
pip install -r requirements.txt

# Créer les dossiers nécessaires
echo -e "\n${YELLOW} Création des dossiers...${NC}"
mkdir -p models
mkdir -p data
mkdir -p mlruns
mkdir -p tests
mkdir -p api
mkdir -p src

# Créer les fichiers __init__.py
touch src/__init__.py
touch api/__init__.py
touch tests/__init__.py

echo -e "\n${GREEN}Installation terminée avec succès!${NC}"
echo -e "\n${YELLOW}Prochaines étapes:${NC}"
echo "1. Activer l'environnement: source venv/bin/activate"
echo "2. Entraîner le modèle: python src/train.py"
echo "3. Lancer MLflow: mlflow ui"
echo "4. Lancer l'API: python api/app.py"
echo ""