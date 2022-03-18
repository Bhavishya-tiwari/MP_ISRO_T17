

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MP_ISRO_T17.settings')

application = get_wsgi_application()
