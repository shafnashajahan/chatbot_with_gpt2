#pip install gtts (google text to speech)
#sudo apt-get install mpg321 (media player to play the audio)


import subprocess
from gtts import gTTS
import os
import playsound

mytext = 'Wellcome to the tarot card'
language = 'en'

myobj = gTTS(text=mytext, lang=language)
fpath = "C:\\Users\\user\\Desktop\\demo_backend\\Voice_chat\\welcome.mp3"
print(os.path.exists(fpath))
myobj.save(fpath)
print(os.path.exists(fpath))
#subprocess.Popen(['mpg123', '-C', fpath])
playsound.playsound(fpath, True)

