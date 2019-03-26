from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

from django.shortcuts import render
from django.template import loader
from django import forms
from django.http import HttpResponseRedirect
import datetime

def index(request):
    template = loader.get_template('mainsite/index.html')
    context = {
        'latest_question_list': "asd",
    }
    return HttpResponse(template.render(context, request))