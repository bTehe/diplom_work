from flask_cors import CORS
import logging

cors = CORS()

def logging_setup(app):
    handler = logging.FileHandler('logs/app.log')
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)