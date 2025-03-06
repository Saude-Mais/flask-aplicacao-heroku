import logging
from flask import Flask, request, jsonify
import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Carregar o modelo treinado
model = keras.models.load_model('models/modelTuberculose.keras')

# Função para pré-processar a imagem
def preprocess_image(img):
    img = img.resize((320, 320))  # Ajuste o tamanho conforme o necessário
    img = np.array(img) / 255.0  # Normalização
    img = np.expand_dims(img, axis=0)  # Adiciona a dimensão do batch
    return img

logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar se um arquivo foi enviado
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        file = request.files["file"]  # Pega o arquivo enviado

        # Converter o arquivo para uma imagem
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((320, 320))  # Ajuste para o tamanho esperado pelo modelo
        logging.info("Preparando a imagem...")
        image = np.array(image) / 255.0  # Normalizar os valores da imagem
        image = np.expand_dims(image, axis=0)  # Adicionar dimensão do batch

        # Fazer a predição
        logging.info("Fazendo a predição...")
        prediction = model.predict(image)
        predicted_class = prediction

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
