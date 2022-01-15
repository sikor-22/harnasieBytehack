from pyexpat import model
from turtle import title
from django.db import models
# Create your models here.

class Post(models.Model):
    title = models.CharField(max_length=255)
    body = models.TextField()

    def __str__(self) -> str:
        return self.title + ' | ' + 'POST'

