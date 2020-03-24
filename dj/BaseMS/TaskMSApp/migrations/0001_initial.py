# Generated by Django 2.2.4 on 2020-03-24 09:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('baseapp_1', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='PrimaryLevel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('primarylevel', models.CharField(default='一般', max_length=20, verbose_name='优先级')),
            ],
        ),
        migrations.CreateModel(
            name='StateCode',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('statedetails', models.CharField(default='未完成', max_length=20, verbose_name='状态')),
            ],
        ),
        migrations.CreateModel(
            name='Task',
            fields=[
                ('tuuid', models.AutoField(editable=False, primary_key=True, serialize=False, verbose_name='唯一标识')),
                ('taskname', models.CharField(max_length=20, unique=True, verbose_name='任务名')),
                ('comment', models.CharField(default='123456', max_length=200, verbose_name='评价')),
                ('startTime', models.DateTimeField(auto_now=True, verbose_name='开始时间')),
                ('endTime', models.DateTimeField(blank=True, editable=False, null=True, verbose_name='结束时间')),
                ('primary', models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='primarylevel_id', to='taskmsapp.PrimaryLevel', verbose_name='优先级')),
                ('starter', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='user_id', to='baseapp_1.User', verbose_name='发起人')),
                ('state', models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='statecode_id', to='taskmsapp.StateCode', verbose_name='完成状态')),
                ('worker', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='worker_uuid', to='baseapp_1.Worker', verbose_name='操作员')),
            ],
            options={
                'verbose_name': '任务',
                'verbose_name_plural': '任务',
            },
        ),
    ]
