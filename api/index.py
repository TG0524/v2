from flask import Flask, request, jsonify
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Create Flask app
app = Flask(__name__)

# Import routes from main app
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app import app as main_app
    
    # Copy routes from main app
    for rule in main_app.url_map.iter_rules():
        if rule.endpoint != 'static':
            app.add_url_rule(
                rule.rule,
                rule.endpoint,
                main_app.view_functions[rule.endpoint],
                methods=rule.methods
            )
except Exception as e:
    # Fallback routes if import fails
    @app.route('/')
    def index():
        return jsonify({
            'message': 'Amazon Ads Automation API',
            'status': 'running',
            'error': f'Main app import failed: {str(e)}'
        })
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'message': 'Serverless function is running'})

# Vercel serverless function handler
def handler(request):
    with app.test_request_context(request.url, method=request.method, data=request.get_data()):
        try:
            response = app.full_dispatch_request()
            return response
        except Exception as e:
            return jsonify({'error': str(e)}), 500