#pip install SpeechRecognition PyAudio
import speech_recognition as sr

# initialise recognizer
r = sr.Recognizer()  # initialise recognizer
with sr.Microphone() as source:  # mention source it will be either microphone or audio
    print("Speak Anything :")
    audio = r.listen(source)     #listen to the source
    try:
        text = r.recongize_google(audio)  # use recognizer to convert audio into text
        print("User : {}".format(text))
    except:
        print("Sorry could not recognize your voice")   #if voice is not recognized except will be print.