# pdfchat/models.py
from django.db import models

class PDFDocument(models.Model):
    file = models.FileField(upload_to='pdfs/')
    processed = models.BooleanField(default=False)

    def __str__(self):
        return self.file.name
