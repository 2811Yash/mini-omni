import speech_recognition as sr
import pyttsx3
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
os.getenv("GOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
# Initialize the recognizer
recognizer = sr.Recognizer()
voice=pyttsx3.init()
close="close"

def omni(text):
    template = "give the answer of this question {text}"
    prompt = PromptTemplate(input_variables=["text"],
                            template=template)
    
    chain=LLMChain(llm=llm,prompt=prompt,verbose=True)
    result=chain(text)
    return result


st.title("omni chatgpt-4o")
# # st.write("on omni")
if (st.button("📝  Give text input")):
    que=st.text_input("type your question")
    res1=omni(que)["text"]
    st.write(res1)
    if (st.button("🔊")):
        voice.say(res1)
    voice.runAndWait()
st.write("OR")
if (st.button("🎙️  Give the voice command")):
    while 1:
        with sr.Microphone() as source:
            st.write("Speak something...")
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source)

            try:
                text = recognizer.recognize_google(audio_data)
                print("Converted Speech:")
                st.write(text)
                res=omni(text)["text"]
                st.write(res)
                voice.say(res)
                voice.runAndWait()

            except sr.UnknownValueError:
                st.write("Speech Recognition could not understand audio")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")
