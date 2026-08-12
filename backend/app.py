# -*- coding: utf-8 -*-
"""
SepsisGuard AI - Central Hospital Telemetry Server v3.0 (Phase 10 / Phase 12 Architecture)
Lightweight Application Factory & Service Entry Point with Structured Logging.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from backend.config import FLASK_SECRET_KEY, API_KEY, allowed_origins, PREDICT_RATE_LIMIT, logger
from backend.api.predict import predict_bp, predict, limiter
from backend.api.patients import patients_bp
from backend.api.sockets import register_socket_events
from backend.validation.schemas import sanitize

def create_app():
    """
    Application Factory: Initializes Flask, configures extensions (CORS, Limiter, SocketIO),
    registers blueprints, and binds real-time socket handlers.
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = FLASK_SECRET_KEY
    app.config['RATELIMIT_ENABLED'] = True

    # 1. Explicit CORS Allow-List (No Wildcard)
    CORS(app, resources={r"/*": {"origins": allowed_origins}})

    # 2. Socket.IO Setup
    socketio = SocketIO(
        app,
        cors_allowed_origins=allowed_origins,
        async_mode='threading',
        logger=False,
        engineio_logger=False,
        ping_timeout=60,
        ping_interval=25
    )

    # 3. Rate Limiter Setup
    limiter.init_app(app)

    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({
            "error": "Too Many Requests",
            "message": "Rate limit exceeded. Please try again later."
        }), 429

    # 4. Register API Blueprints & Socket Events
    app.register_blueprint(predict_bp)
    app.register_blueprint(patients_bp)
    register_socket_events(socketio)

    return app, socketio, limiter

# Instantiate main app and socketio for WSGI (Gunicorn) & test suites
app, socketio, limiter = create_app()

# Re-export helper symbols for test suite backwards compatibility
from backend.ml.inference import run_ml_pipeline
from backend.services.copilot import _local_synthesis

if __name__ == '__main__':
    logger.info("=" * 55)
    logger.info("  SepsisGuard AI v3.0 - ICU Intelligence Ecosystem")
    logger.info("  http://localhost:5000")
    logger.info("=" * 55)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
