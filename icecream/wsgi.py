import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'icecream.settings')  # Match your project name
application = get_wsgi_application()