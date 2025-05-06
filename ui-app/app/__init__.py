# app/__init__.py
from flask import Flask
from .extensions import cors, logging_setup
from .blueprints.main.routes import main_bp
from .blueprints.data_ingest.routes import data_ingest_bp
from .blueprints.preprocessing.routes import preprocess_bp
from .blueprints.training.routes import training_bp
from .blueprints.inference.routes import inference_bp
from .blueprints.monitoring.routes    import monitoring_bp
from .blueprints.reports.routes       import reports_bp

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object('config')
    app.config.from_pyfile('config.py', silent=True)

    cors.init_app(app)
    logging_setup(app)

    app.register_blueprint(main_bp)
    app.register_blueprint(data_ingest_bp)
    app.register_blueprint(preprocess_bp) 
    app.register_blueprint(training_bp)
    app.register_blueprint(inference_bp)
    app.register_blueprint(monitoring_bp)
    app.register_blueprint(reports_bp)

    return app