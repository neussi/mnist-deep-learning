#!/bin/bash

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE} CONSTRUCTION ET DÉMARRAGE DOCKER - MNIST API${NC}"
echo -e "${BLUE}======================================================================${NC}\n"

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
    echo -e "${RED} Docker n'est pas installé${NC}"
    echo -e "${YELLOW} Installez Docker: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

echo -e "${GREEN} Docker détecté: $(docker --version)${NC}\n"

# Vérifier que le modèle existe
if [ ! -f "models/mnist_model.h5" ]; then
    echo -e "${RED} Modèle non trouvé: models/mnist_model.h5${NC}"
    echo -e "${YELLOW} Veuillez d'abord entraîner le modèle:${NC}"
    echo -e "${YELLOW}   python src/train.py${NC}\n"
    exit 1
fi

echo -e "${GREEN} Modèle trouvé: models/mnist_model.h5${NC}\n"

# Nettoyer les anciens conteneurs
echo -e "${YELLOW} Nettoyage des anciens conteneurs...${NC}"
docker stop mnist-api 2>/dev/null || true
docker rm mnist-api 2>/dev/null || true

# Construire l'image
echo -e "\n${YELLOW} Construction de l'image Docker...${NC}"
docker build -t mnist-api:latest . --no-cache

if [ $? -ne 0 ]; then
    echo -e "${RED} Erreur lors de la construction de l'image${NC}"
    exit 1
fi

echo -e "${GREEN} Image construite avec succès${NC}\n"

# Afficher la taille de l'image
IMAGE_SIZE=$(docker images mnist-api:latest --format "{{.Size}}")
echo -e "${BLUE} Taille de l'image: ${IMAGE_SIZE}${NC}\n"

# Lancer le conteneur
echo -e "${YELLOW} Démarrage du conteneur...${NC}"
docker run -d \
    --name mnist-api \
    -p 5000:5000 \
    -v "$(pwd)/models:/app/models:ro" \
    --restart unless-stopped \
    mnist-api:latest

if [ $? -ne 0 ]; then
    echo -e "${RED} Erreur lors du démarrage du conteneur${NC}"
    exit 1
fi

echo -e "${GREEN} Conteneur démarré avec succès${NC}\n"

# Attendre que l'API soit prête
echo -e "${YELLOW} Attente du démarrage de l'API...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo -e "${GREEN} API opérationnelle!${NC}\n"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -e "${YELLOW}   Tentative $RETRY_COUNT/$MAX_RETRIES...${NC}"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED} Timeout: L'API n'a pas démarré${NC}"
    echo -e "${YELLOW} Vérifiez les logs: docker logs mnist-api${NC}\n"
    exit 1
fi

# Afficher les informations
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN} DÉPLOIEMENT RÉUSSI!${NC}"
echo -e "${BLUE}======================================================================${NC}\n"

echo -e "${YELLOW} Informations du conteneur:${NC}"
docker ps --filter "name=mnist-api" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo -e "\n${YELLOW} Endpoints disponibles:${NC}"
echo -e "   ${BLUE}http://localhost:5000/${NC}          - Page d'accueil"
echo -e "   ${BLUE}http://localhost:5000/health${NC}    - Health check"
echo -e "   ${BLUE}http://localhost:5000/predict${NC}   - Prédiction"
echo -e "   ${BLUE}http://localhost:5000/model/info${NC} - Info modèle"

echo -e "\n${YELLOW} Tester l'API:${NC}"
echo -e "   ${BLUE}curl http://localhost:5000/health${NC}"
echo -e "   ${BLUE}python api/test_api.py${NC}"

echo -e "\n${YELLOW} Commandes utiles:${NC}"
echo -e "   ${BLUE}docker logs mnist-api${NC}           - Voir les logs"
echo -e "   ${BLUE}docker stop mnist-api${NC}           - Arrêter le conteneur"
echo -e "   ${BLUE}docker restart mnist-api${NC}        - Redémarrer"
echo -e "   ${BLUE}docker exec -it mnist-api bash${NC}  - Shell dans le conteneur"

echo -e "\n${BLUE}======================================================================${NC}\n"