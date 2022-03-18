

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MP_ISRO_T17.settings')

application = get_asgi_application()
