"""
Top-level package initialises the Flask application factory so external
callers can simply `import webapp` then `webapp.create_app()`.
"""
from .app_factory import create_app  # re-export

__all__ = ["create_app"]
