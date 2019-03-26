
import string
import random

def takeInput(data):
    asdf = id_generator() + " " + str(data)
    return asdf

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

