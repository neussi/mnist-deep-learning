#!/bin/bash
set -e

echo " Démarrage du conteneur MNIST API..."

# Vérifier que le modèle existe
if [ ! -f "$MODEL_PATH" ]; then
    echo " ERREUR: Modèle non trouvé à $MODEL_PATH"
    echo " Veuillez monter le volume avec le modèle ou entraîner le modèle"
    exit 1
fi

echo "Modèle trouvé: $MODEL_PATH"

# Vérifier les permissions
if [ ! -r "$MODEL_PATH" ]; then
    echo " ERREUR: Pas de permission de lecture sur le modèle"
    exit 1
fi

echo " Permissions OK"

# Créer le dossier logs s'il n'existe pas
mkdir -p /app/logs

echo " Démarrage de l'API Flask..."

# Lancer l'application
exec python api/app.py