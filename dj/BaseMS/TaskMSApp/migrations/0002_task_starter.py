# Generated by Django 2.2.4 on 2020-03-18 09:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('TaskMSApp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='task',
            name='starter',
            field=models.CharField(default='', max_length=20, verbose_name='发起人'),
        ),
    ]
