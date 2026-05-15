from django.db import models
from django.contrib.auth.models import User

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    prediction_type = models.CharField(max_length=50)
    input_data = models.JSONField()
    prediction = models.IntegerField()
    prediction_label = models.CharField(max_length=100, blank=True)
    probabilities = models.JSONField()
    advice = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction_type} prediction for {self.user.username} at {self.created_at}"

class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message_type = models.CharField(max_length=10, choices=(('sent', 'Sent'), ('received', 'Received')))
    message_text = models.TextField()
    prediction_history = models.ForeignKey(PredictionHistory, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.message_type} message by {self.user.username} at {self.created_at}"