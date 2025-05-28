from flask import Flask
from flask_restful import Api
from webapp.apis.v1.analysis_routes import initialize_analysis_routes
from webapp.tasks import celery_app # Import the celery_app instance

def create_app(config_name=None): # config_name can be 'development', 'production' etc.
    app = Flask(__name__)

    # --- Configuration ---
    # Load configuration from a config file or environment variables
    # Example: app.config.from_object('webapp.config.DevelopmentConfig')
    app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

    # Initialize Celery
    celery_app.conf.update(broker_url=app.config['CELERY_BROKER_URL'],
                           result_backend=app.config['CELERY_RESULT_BACKEND'])
    # To make celery tasks aware of Flask app context if needed (e.g., for db access within tasks)
    # class FlaskTask(celery_app.Task):
    #     def __call__(self, *args, **kwargs):
    #         with app.app_context():
    #             return self.run(*args, **kwargs)
    # celery_app.Task = FlaskTask
    
    # Initialize Flask-RESTful API
    api_bp_v1 = Api(app, prefix='/api/v1') # Register API with prefix
    initialize_analysis_routes(api_bp_v1)

    # --- Basic Error Handlers ---
    @app.errorhandler(404)
    def not_found_error(error):
        return {"error": "Not Found", "message": str(error)}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal Server Error", "message": str(error)}, 500
        
    @app.route('/')
    def hello():
        return "Ncorr Python WebApp is running!"

    return app

if __name__ == '__main__':
    # This is for development only. For production, use Gunicorn.
    # Example: gunicorn --bind 0.0.0.0:5000 "webapp.app:create_app()"
    # To run Celery worker: celery -A webapp.tasks.celery_app worker -l info
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)