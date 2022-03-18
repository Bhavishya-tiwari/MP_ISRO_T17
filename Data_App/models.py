from django.db import models

# Create your models here.
class files(models.Model):
    uniqueid = models.AutoField(primary_key=True),
    Name = models.TextField(default=""),
    csvfile = models.FileField(),
    csvfile = models.ImageField(upload_to=''),

    def __str__(self):
        return f'{self.user.uniqueid}'