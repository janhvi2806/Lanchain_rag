from django.db import models

# Create your models here.
import uuid
from django.db import models
from django.conf import settings


class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField('Document', upload_to=settings.FILE_STORAGE_DIR)
