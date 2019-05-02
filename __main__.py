from flask import Flask
from flasgger import Swagger
from query import query_page
from scape import scape_page

if __name__ == "__main__":
    app = Flask(__name__)

    app.register_blueprint(query_page, url_prefix="/query")
    app.register_blueprint(scape_page, url_prefix="/scape")
    # CORS(app)
    Swagger(app)
    app.run(debug=True, port=5000)
