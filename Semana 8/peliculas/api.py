from flask import Flask, request
from flask_restx import Api, Resource, fields, reqparse
import joblib
from predict import predict_proba
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
import pandas as pd
app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Car Price Prediction API',
    description='Car Price Prediction API'
)
ns = api.namespace('predict', description='Car Price')
# Definición de campos de respuesta
response_model = api.model('Response', {
    'Action': fields.Float,
    'Adventure': fields.Float,
    'Animation': fields.Float,
    'Biography': fields.Float,
    'Comedy': fields.Float,
    'Crime': fields.Float,
    'Documentary': fields.Float,
    'Drama': fields.Float,
    'Family': fields.Float,
    'Fantasy': fields.Float,
    'Film-Noir': fields.Float,
    'History': fields.Float,
    'Horror': fields.Float,
    'Music': fields.Float,
    'Musical': fields.Float,
    'Mystery': fields.Float,
    'News': fields.Float,
    'Romance': fields.Float,
    'Sci-Fi': fields.Float,
    'Short': fields.Float,
    'Sport': fields.Float,
    'Thriller': fields.Float,
    'War': fields.Float,
    'Western': fields.Float
})
# Definición de argumentos para la API
parser = reqparse.RequestParser()
parser.add_argument('Plot', type=str, required=True, help='Sinopsis de la pelicula')

# Definición de la clase para la API
@ns.route('/')
class CarPriceApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(response_model)
    def post(self):
        args = parser.parse_args()
        # Llama a la función predict_price para realizar la predicción
        prediction = predict_proba(args['Plot'])
        # Devuelve el resultado como respuesta JSON
        return prediction, 200
# Inicia el servidor Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)