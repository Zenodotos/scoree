from django import forms
from .models import Visit, Patient

class VisitForm(forms.ModelForm):
    class Meta:
        model = Visit
        fields = ['visit_date', 'systolic_pressure', 'cholesterol_total', 
                 'cholesterol_hdl', 'hba1c', 'egfr']
        widgets = {
            'visit_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'systolic_pressure': forms.NumberInput(attrs={'class': 'form-control'}),
            'cholesterol_total': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'cholesterol_hdl': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'hba1c': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'egfr': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
        }

class PatientSmokingForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['smoking_status']
        widgets = {
            'smoking_status': forms.Select(attrs={'class': 'form-control'})
        }