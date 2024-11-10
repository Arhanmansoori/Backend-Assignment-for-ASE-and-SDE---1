# pdfchat/urls.py
from django.urls import path
from .views import PDFUploadView, QuestionAnswerView

urlpatterns = [
    path('upload_pdf/', PDFUploadView.as_view(), name='upload_pdf'),
    path('ask_question/', QuestionAnswerView.as_view(), name='ask_question'),
]
