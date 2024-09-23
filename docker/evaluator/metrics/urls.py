from django.urls import path
from . import views

urlpatterns = [
    path("features", views.features, name="features"),
    path("hf_evaluator", views.hf_evaluator, name="huggingface_evaluator"),

]