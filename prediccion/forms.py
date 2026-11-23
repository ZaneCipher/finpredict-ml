from django import forms

class UploadCSVForm(forms.Form):
    archivo_csv = forms.FileField()

class PrediccionForm(forms.Form):
    x1 = forms.FloatField(min_value=0)
    x2 = forms.FloatField(min_value=0)
    x3 = forms.FloatField(min_value=0)
    x4 = forms.FloatField(min_value=0)
