from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

from django.shortcuts import render
from django.template import loader
from django import forms
from django.http import HttpResponseRedirect
import datetime

def index(request):
    template = loader.get_template('myapp/home.html')
    context = {
        'latest_question_list': "asd",
    }
    Logger("Myapp started")
    return HttpResponse(template.render(context, request))

def Logger(msg):
    # Log an error message
    with open("E:\\python\\jangoD invisible\\myproject5\\myproject3\\myapp\\nlu\\logger.txt", "a") as myfile:
        myfile.write(str(datetime.datetime.now()) +" "+ str(msg) + "\n")

class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)

from .nlu import ml
from .nlu import natural_language_processing
import json

# else :
#   return render_to_response('ajax_test.html', locals())

def get_name(request):
    Logger("inside get_name")
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        Logger("inside POST")
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            Logger("inside POST valid")
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            # return HttpResponseRedirect('/myapp/thanks')

            modified = str(request.POST['your_name']);

            # return HttpResponse(modified)
            # template = loader.get_template('myapp/home.html')
            # context = {
            #     'getName': newData,
            # }
            # return HttpResponse(template.render(context, request))
            # if request.method == 'POST' and request.is_ajax():

            name = request.POST.get('your_name')
            key = request.POST.get('csrfmiddlewaretoken')
            Logger("Name  ->" + str(name) + " : " + str(key) + " <- ")
            newData = natural_language_processing.takeInput(name)

            Logger("count " +str(newData) + " <- ")
            Logger("Name  ->" + str(name) + " : " + str(key)+ " : "+str(newData)+" <- ")
            Logger("new Data  ->" + str(newData[0][0]) + " : " + str(newData[1]) + " : " + str(newData[2]) +" : "+str(newData[3])+" <- ")
            intent_nostr = str(newData[0][0])
            log_probstr =  str(newData[1])


            probstr = newData[2]
            const_arr = 0
            for i in range(len(probstr[0])):
                if probstr[0][i] == 1:
                    const_arr = probstr[0][i]
            if const_arr != 1:
                probstr = "intent not found error"
            else:
                probstr = str(newData[2])


            personstr = str(newData[3])

            return HttpResponse(json.dumps({'intent_no': intent_nostr, 'log_prob': log_probstr , 'prob': probstr, 'person': personstr}), content_type="application/json")



    # if a GET (or any other method) we'll create a blank form
    else:
        Logger("inside POST fail")
        form = NameForm()
    return render(request, 'myapp/name.html', {'form': form})


def thanks(request):
    return HttpResponse("Thanks")

def train(request):
    template = loader.get_template('myapp/dashboard.html')
    context = {
        'latest_question_list': "asd",
    }
    Logger("Myapp started")
    return HttpResponse(template.render(context, request))
