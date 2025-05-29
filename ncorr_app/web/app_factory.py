"""
Flask application factory – wires configuration, Celery and Blueprints.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from celery import Celery
from flask import Flask

from webapp.config import cfg


def _make_celery(app: Flask) -> Celery:
    celery = Celery(
        app.import_name,
        broker=cfg.CELERY_BROKER_URL,
        backend=cfg.CELERY_RESULT_BACKEND,
    )
    celery.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
    )

    # Ensure Celery tasks use Flask app context
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):  # noqa: D401
            with app.app_context():
                return super().__call__(*args, **kwargs)

    celery.Task = ContextTask
    return celery


def create_app(config_name: str = "default") -> Flask:
    app = Flask(__name__)
    app.config.from_object(cfg)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s\t%(name)s\t%(message)s",
    )

    # Celery instance attached to app for easy access
    app.extensions = getattr(app, "extensions", {})
    app.extensions["celery"] = _make_celery(app)

    # Register blueprints
    from webapp.apis.v1.routes_analysis import bp_analysis  # pylint: disable=import-error
    app.register_blueprint(bp_analysis)

    @app.route("/health")
    def _health():
        return {"status": "ok"}

    return app
