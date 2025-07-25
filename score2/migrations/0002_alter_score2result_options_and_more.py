# Generated by Django 5.2.4 on 2025-07-25 12:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('patients', '0002_patient_smoking_status'),
        ('score2', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='score2result',
            options={'ordering': ['-score_value', '-created_at']},
        ),
        migrations.AlterUniqueTogether(
            name='score2result',
            unique_together={('patient', 'visit')},
        ),
        migrations.AddField(
            model_name='score2result',
            name='data_source',
            field=models.CharField(choices=[('visit', 'Dane z wizyty'), ('previous_visit', 'Dane z poprzednich wizyt'), ('median', 'Mediana dla grupy wiekowej'), ('mixed', 'Dane mieszane')], default='visit', max_length=20),
        ),
    ]
