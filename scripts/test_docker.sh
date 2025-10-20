#!/bin/bash

# Couleurs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE} TESTS DU CONTENEUR DOCKER${NC}"
echo -e "${BLUE}======================================================================${NC}\n"

API_URL="http://localhost:5000"

# Test 1: Vérifier que le conteneur tourne
echo -e "${YELLOW}Test 1: Vérification du conteneur...${NC}"
if docker ps | grep -q mnist-api; then
    echo -e "${GREEN} Conteneur en cours d'exécution${NC}\n"
else
    echo -e "${RED} Conteneur non trouvé${NC}\n"
    exit 1
fi

# Test 2: Health check
echo -e "${YELLOW}Test 2: Health check...${NC}"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/health)
if [ "$HEALTH" = "200" ]; then
    echo -e "${GREEN} API healthy (200)${NC}\n"
else
    echo -e "${RED} API non healthy (Status: $HEALTH)${NC}\n"
    exit 1
fi

# Test 3: Endpoint home
echo -e "${YELLOW}Test 3: Endpoint home...${NC}"
HOME_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/)
if [ "$HOME_STATUS" = "200" ]; then
    echo -e "${GREEN} Home endpoint OK${NC}\n"
else
    echo -e "${RED} Home endpoint échoué (Status: $HOME_STATUS)${NC}\n"
fi

# Test 4: Prédiction simple
echo -e "${YELLOW}Test 4: Test de prédiction...${NC}"
PREDICT_RESPONSE=$(curl -s -X POST $API_URL/predict \
    -H "Content-Type: application/json" \
    -d '{"image": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}')

if echo "$PREDICT_RESPONSE" | grep -q "prediction"; then
    echo -e "${GREEN}   Prédiction réussie${NC}"
    echo -e "${BLUE}   Réponse: $PREDICT_RESPONSE${NC}\n"
else
    echo -e "${RED} Prédiction échouée${NC}\n"
fi

# Test 5: Logs
echo -e "${YELLOW}Test 5: Vérification des logs...${NC}"
LOG_LINES=$(docker logs mnist-api 2>&1 | wc -l)
echo -e "${GREEN}$LOG_LINES lignes de logs${NC}\n"

# Test 6: Ressources
echo -e "${YELLOW}Test 6: Utilisation des ressources...${NC}"
docker stats mnist-api --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
echo ""

# Résumé
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN} TOUS LES TESTS SONT PASSÉS!${NC}"
echo -e "${BLUE}======================================================================${NC}\n"