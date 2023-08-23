from google.cloud import texttospeech

# Create a client for the Google Cloud Text-to-Speech API
client = texttospeech.TextToSpeechClient()

# Input text to be converted to speech
input_text = "Hello, this is a test of the Text-to-Speech API."

# Set the text synthesis input
input_text_synthesis = texttospeech.SynthesisInput(text=input_text)

# Select the voice and audio configuration
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
)
audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

# Perform text-to-speech synthesis
response = client.synthesize_speech(input_text_synthesis, voice, audio_config)

# Save the synthesized audio to a file
output_audio_file = "output_audio.mp3"
with open(output_audio_file, "wb") as out_audio:
    out_audio.write(response.audio_content)
    print(f"Audio content written to file {output_audio_file}")
