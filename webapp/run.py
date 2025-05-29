"""
CLI entry-point for the development Flask server.
"""
from webapp.app_factory import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
