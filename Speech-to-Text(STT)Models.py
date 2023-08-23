from google.cloud import speech

# Create a client for the Google Cloud Speech-to-Text API
client = speech.SpeechClient()

# Path to the audio file you want to transcribe
audio_file_path = 'path_to_audio_file.wav'

# Load the audio file
with open(audio_file_path, 'rb') as audio_file:
    audio_content = audio_file.read()

# Set the configuration for the speech recognition
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # Replace with appropriate encoding
    sample_rate_hertz=16000,  # Replace with the sample rate of your audio file
    language_code="en-US",   # Replace with the desired language code
)

# Perform the speech recognition
response = client.recognize(config=config, audio=speech.RecognitionAudio(content=audio_content))

# Print the transcribed text
for result in response.results:
    print("Transcript: {}".format(result.alternatives[0].transcript))
