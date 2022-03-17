from django.db import models

# Create your models here.
class files(models.Model):
    uniqueid = models.AutoField(primary_key=True),
    Name = models.TextField(default=""),
    timeStamp = models.FileField(),

    def __str__(self):
        return f'{self.user.uniqueid}'