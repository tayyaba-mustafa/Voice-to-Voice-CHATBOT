# Import libraries
import whisper
import os
from groq import Groq
from gtts import gTTS
import gradio as gr

# Load Whisper model for transcription
whisper_model = whisper.load_model("small")  # Adjust model size as needed

GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)

# Function to transcribe audio using Whisper
def transcribe_audio(file):
    try:
        transcription = whisper_model.transcribe(file)
        return transcription['text']
    except Exception as e:
        return f"Transcription error: {str(e)}"

# Function to get a chat response from the LLM using Groq's API
def get_chat_response(transcription):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": transcription}],
            model="llama3-8b-8192"  # Specify the model you intend to use
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Chat response error: {str(e)}"

# Function to convert text response to audio using gTTS
def text_to_speech(response_text):
    try:
        tts = gTTS(response_text)
        audio_file = "response_audio.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        return f"Text-to-speech error: {str(e)}"

# Function that combines the entire chatbot flow
def chatbot(audio):
    transcription = transcribe_audio(audio)
    if "error" in transcription:
        return transcription, None  # Return the error message and no audio

    response_text = get_chat_response(transcription)
    if "error" in response_text:
        return response_text, None  # Return the error message and no audio

    response_audio = text_to_speech(response_text)
    return response_text, response_audio

# Set up Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Audio(type="filepath"),  # Microphone input
    outputs=[gr.Textbox(), gr.Audio(type="filepath")],
    live=True
)

# Launch the Gradio app
iface.launch()
