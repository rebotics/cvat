# Generated by Django 3.2.15 on 2023-08-25 05:08

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('rebotics', '0006_alter_galleryimportprogress_url'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='galleryimportprogress',
            name='url',
        ),
    ]
