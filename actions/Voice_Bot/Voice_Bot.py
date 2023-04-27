## Run this command in terminal  before executing this program
## rasa run -m models --endpoints endpoints.yml --port 5002 --credentials credentials.yml
## and also run this in seperate terminal
## rasa run actions

import requests
import speech_recognition as sr     # import the library
import subprocess
from gtts import gTTS
import os
import playsound


#sender = input("What is your name?\n")

bot_message = ""
message=""

r = requests.post('http://localhost:5002/webhooks/rest/webhook', json={"message": "Hello"})

print("Bot says, ",end=' ')
for i in r.json():
    bot_message = i['text']
    print(f"{bot_message}")
    
myobj = gTTS(text=bot_message)
myobj.save("C:\\Users\\user\\Desktop\\demo_backend\\Voice\\welcome.mp3")
#print('saved')
# Playing the converted file
#subprocess.call(['mpg321', "C:\\Users\\user\\Desktop\\demo_backend\\Voice\\welcome.mp3", '--play-and-exit'])
playsound.playsound('C:\\Users\\user\\Desktop\\demo_backend\\Voice\\welcome.mp3', True)    
    
def speech_to_text():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak Anything :")
            audio = r.listen(source)
            try:
                message = r.recognize_google(audio)  # use recognizer to convert our audio into text part.
                print("You said : {}".format(message))
                # print(text)
            except:
                 print("Sorry could not recognize your voice")
            # In case of voice not recognized  clearly
            return message
        
while bot_message != "Bye" or bot_message!='thanks':   
    if len(message)==0:
        continue
    print("Sending message now...")        
        #print(speech_to_text())       
    r = requests.post('http://localhost:5002/webhooks/rest/webhook', json={"message": message})
    print("Bot says, ",end=' ')
    for i in r.json():
        bot_message = i['text']
        print(f"{bot_message}")
    myobj = gTTS(text=bot_message)
    myobj.save("C:\\Users\\user\\Desktop\\welcome.mp3")
    playsound.playsound("C:\\Users\\user\\Desktop\\welcome.mp3", True)
