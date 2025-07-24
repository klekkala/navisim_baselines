#!/bin/bash

ENV_NAME="navisim_env"

echo "🚀 Creating virtual environment: $ENV_NAME"
python3 -m venv $ENV_NAME

echo "🐍 Activating virtual environment..."
source $ENV_NAME/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing required packages..."
pip install stable-baselines3 mani-skill matplotlib torch pillow
pip install hydra-core omegaconf termcolor

echo "✅ Environment setup complete!"
echo "👉 To activate it later, run: source $ENV_NAME/bin/activate"
