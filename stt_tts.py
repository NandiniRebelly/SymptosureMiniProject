"""
Speech-to-Text and Text-to-Speech module for symptoms checker project.
Provides both cloud and local speech processing capabilities.
"""

import os
import io
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Union
import wave


class SpeechProcessor:
    """
    Handles speech-to-text and text-to-speech processing with cloud and local backends.
    
    Cloud Mode: Uses Google Cloud Speech and Text-to-Speech APIs (requires credentials)
    Local Mode: Uses SpeechRecognition and gTTS libraries (free but limited)
    """
    
    def __init__(self):
        """
        Initialize speech processor with appropriate backends based on environment.
        """
        self.google_cloud_available = self._check_google_cloud_credentials()
        self.local_stt_available = self._check_speechrecognition_availability()
        self.local_tts_available = self._check_gtts_availability()
        
        # Initialize cloud clients
        self.speech_client = None
        self.tts_client = None
        
        if self.google_cloud_available:
            self._init_google_cloud_clients()
            print("✓ Google Cloud Speech services initialized (Production Mode)")
        else:
            print("⚠️ Google Cloud credentials not found - using local fallbacks")
        
        if self.local_stt_available:
            print("✓ SpeechRecognition available (Local STT)")
        else:
            print("⚠️ SpeechRecognition not available")
        
        if self.local_tts_available:
            print("✓ gTTS available (Local TTS)")
        else:
            print("⚠️ gTTS not available")
    
    def _check_google_cloud_credentials(self) -> bool:
        """
        Check if Google Cloud credentials are available.
        
        Returns:
            True if GOOGLE_APPLICATION_CREDENTIALS environment variable is set
        """
        return os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is not None
    
    def _check_speechrecognition_availability(self) -> bool:
        """
        Check if SpeechRecognition library is available.
        
        Returns:
            True if SpeechRecognition can be imported
        """
        try:
            import speech_recognition as sr
            return True
        except ImportError:
            return False
    
    def _check_gtts_availability(self) -> bool:
        """
        Check if gTTS library is available.
        
        Returns:
            True if gTTS can be imported
        """
        try:
            from gtts import gTTS
            return True
        except ImportError:
            return False
    
    def _init_google_cloud_clients(self) -> None:
        """
        Initialize Google Cloud Speech and Text-to-Speech clients.
        This is the PRODUCTION mode - requires valid Google Cloud credentials.
        
        Rate Limits:
        - Speech-to-Text: 60 requests per minute (free tier)
        - Text-to-Speech: 1 million characters per month (free tier)
        """
        try:
            from google.cloud import speech
            from google.cloud import texttospeech
            
            self.speech_client = speech.SpeechClient()
            self.tts_client = texttospeech.TextToSpeechClient()
            
            print("✓ Google Cloud Speech and TTS clients initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Google Cloud clients: {e}")
            self.speech_client = None
            self.tts_client = None
    
    def speech_to_text_cloud(self, audio_bytes: bytes, language_code: str = 'en-US') -> str:
        """
        Convert speech to text using Google Cloud Speech-to-Text API.
        
        This is the PRODUCTION mode with high accuracy and reliability.
        
        Rate Limits:
        - Free tier: 60 requests per minute
        - Paid tier: Higher limits available
        
        Args:
            audio_bytes: Raw audio data (WAV, FLAC, or MP3 format)
            language_code: Language code (e.g., 'en-US', 'hi-IN', 'pa-IN')
            
        Returns:
            Transcribed text
            
        Raises:
            ValueError: If cloud client not available or transcription fails
        """
        if not self.speech_client:
            raise ValueError("Google Cloud Speech client not available. Check credentials.")
        
        if not audio_bytes:
            raise ValueError("Audio data is empty")
        
        try:
            # Configure audio settings
            audio = speech.RecognitionAudio(content=audio_bytes)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,  # Standard sample rate
                language_code=language_code,
                enable_automatic_punctuation=True,
                model='latest_long'  # Best model for longer audio
            )
            
            # Perform speech recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            # Extract transcribed text
            if response.results:
                transcribed_text = response.results[0].alternatives[0].transcript
                return transcribed_text.strip()
            else:
                return ""
                
        except Exception as e:
            raise ValueError(f"Google Cloud Speech-to-Text failed: {e}")
    
    def text_to_speech_cloud(self, text: str, language_code: str = 'en-US') -> Tuple[str, bytes]:
        """
        Convert text to speech using Google Cloud Text-to-Speech API.
        
        This is the PRODUCTION mode with high-quality voices and multiple languages.
        
        Rate Limits:
        - Free tier: 1 million characters per month
        - Paid tier: Higher limits available
        
        Args:
            text: Text to convert to speech
            language_code: Language code (e.g., 'en-US', 'hi-IN', 'pa-IN')
            
        Returns:
            Tuple of (text, audio_bytes)
            
        Raises:
            ValueError: If cloud client not available or synthesis fails
        """
        if not self.tts_client:
            raise ValueError("Google Cloud Text-to-Speech client not available. Check credentials.")
        
        if not text or not text.strip():
            raise ValueError("Text is empty")
        
        try:
            # Configure synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Configure voice parameters
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            
            # Configure audio format
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            # Perform text-to-speech synthesis
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            return text, response.audio_content
            
        except Exception as e:
            raise ValueError(f"Google Cloud Text-to-Speech failed: {e}")
    
    def speech_to_text_local(self, audio_file_path: Union[str, Path]) -> str:
        """
        Convert speech to text using local SpeechRecognition library.
        
        This is the PROTOTYPING mode using Google Web Speech API.
        
        Limitations:
        - Requires internet connection
        - Results vary significantly in quality
        - Limited to shorter audio clips
        - May have connection issues
        - Free but unofficial service
        
        Args:
            audio_file_path: Path to audio file (WAV format recommended)
            
        Returns:
            Transcribed text
            
        Raises:
            ValueError: If SpeechRecognition not available or transcription fails
        """
        if not self.local_stt_available:
            raise ValueError("SpeechRecognition library not available")
        
        if not os.path.exists(audio_file_path):
            raise ValueError(f"Audio file not found: {audio_file_path}")
        
        try:
            import speech_recognition as sr
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(str(audio_file_path)) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)
                # Record audio data
                audio_data = recognizer.record(source)
            
            # Recognize speech using Google Web Speech API
            try:
                # transcribed_text = recognizer.recognize_google(audio_data)
                transcribed_text = recognizer.recognize_google(audio_data, language="en-IN")
                return transcribed_text.strip()
            except sr.UnknownValueError:
                return ""
            except sr.RequestError as e:
                raise ValueError(f"Speech recognition service error: {e}")
                
        except Exception as e:
            raise ValueError(f"Local speech-to-text failed: {e}")
    
    def text_to_speech_local(self, text: str, lang: str = 'en') -> Tuple[str, bytes, str]:
        """
        Convert text to speech using local gTTS library.
        
        This is the PROTOTYPING mode using Google Translate TTS.
        
        Limitations:
        - Requires internet connection
        - Limited voice quality and options
        - Rate limited by Google
        - Free but unofficial service
        
        Args:
            text: Text to convert to speech
            lang: Language code (e.g., 'en', 'hi', 'pa')
            
        Returns:
            Tuple of (text, audio_bytes, temp_file_path)
            
        Raises:
            ValueError: If gTTS not available or synthesis fails
        """
        if not self.local_tts_available:
            raise ValueError("gTTS library not available")
        
        if not text or not text.strip():
            raise ValueError("Text is empty")
        
        try:
            from gtts import gTTS
            
            # # Create gTTS object
            # tts = gTTS(text=text, lang=lang, slow=False)

            gtts_lang_map = {
        "en": "en",
        "hi": "hi",
        "te": "te",
        "pa": "pa",
        "kn": "kn",
        "ta": "ta",
            }
            tts = gTTS(text=text, lang=gtts_lang_map.get(lang, "en"), slow=False)
                
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file_path = temp_file.name
            
            # Save audio to temporary file
            tts.save(temp_file_path)
            
            # Read audio bytes
            with open(temp_file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            return text, audio_bytes, temp_file_path
            
        except Exception as e:
            raise ValueError(f"Local text-to-speech failed: {e}")
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary file created by local TTS.
        
        Args:
            file_path: Path to temporary file to delete
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {file_path}: {e}")
    
    def get_available_languages(self) -> dict:
        """
        Get list of available languages for speech processing.
        
        Returns:
            Dictionary mapping language codes to language names
        """
        # return {
        #     'en': 'English',
        #     'en-US': 'English (US)',
        #     'hi': 'Hindi',
        #     'hi-IN': 'Hindi (India)',
            
        #     'te': 'Telugu',
        #     'te-IN': 'Telugu (India)',
        
        #     'pa': 'Punjabi',
        #     'pa-IN': 'Punjabi (India)',
        #     'es': 'Spanish',
        #     'es-ES': 'Spanish (Spain)',
        #     'fr': 'French',
        #     'fr-FR': 'French (France)',
        #     'de': 'German',
        #     'de-DE': 'German (Germany)',
        #     'it': 'Italian',
        #     'it-IT': 'Italian (Italy)',
        #     'pt': 'Portuguese',
        #     'pt-PT': 'Portuguese (Portugal)',
        #     'ru': 'Russian',
        #     'ru-RU': 'Russian (Russia)',
        #     'ja': 'Japanese',
        #     'ja-JP': 'Japanese (Japan)',
        #     'ko': 'Korean',
        #     'ko-KR': 'Korean (Korea)',
        #     'zh': 'Chinese',
        #     'zh-CN': 'Chinese (Simplified)'
        # }

        return {
    'en': 'English',
    'en-US': 'English (US)',

    'hi': 'Hindi',
    'hi-IN': 'Hindi (India)',

    'te': 'Telugu',
    'te-IN': 'Telugu (India)',

    'pa': 'Punjabi',
    'pa-IN': 'Punjabi (India)',

    'kn': 'Kannada',
    'kn-IN': 'Kannada (India)',

    'ta': 'Tamil',
    'ta-IN': 'Tamil (India)',
}
    
    def get_service_info(self) -> dict:
        """
        Get information about available speech processing services.
        
        Returns:
            Dictionary containing service availability and capabilities
        """
        return {
            'google_cloud_available': self.google_cloud_available,
            'local_stt_available': self.local_stt_available,
            'local_tts_available': self.local_tts_available,
            'speech_client_initialized': self.speech_client is not None,
            'tts_client_initialized': self.tts_client is not None,
            'available_languages': len(self.get_available_languages())
        }


# Global speech processor instance
_speech_processor_instance = None


def get_speech_processor() -> SpeechProcessor:
    """
    Get or create global speech processor instance.
    
    Returns:
        SpeechProcessor instance
    """
    global _speech_processor_instance
    if _speech_processor_instance is None:
        _speech_processor_instance = SpeechProcessor()
    return _speech_processor_instance


def speech_to_text_cloud(audio_bytes: bytes, language_code: str = 'en-US') -> str:
    """
    Convenience function for cloud speech-to-text.
    
    Args:
        audio_bytes: Raw audio data
        language_code: Language code
        
    Returns:
        Transcribed text
    """
    processor = get_speech_processor()
    return processor.speech_to_text_cloud(audio_bytes, language_code)


def text_to_speech_cloud(text: str, language_code: str = 'en-US') -> Tuple[str, bytes]:
    """
    Convenience function for cloud text-to-speech.
    
    Args:
        text: Text to convert to speech
        language_code: Language code
        
    Returns:
        Tuple of (text, audio_bytes)
    """
    processor = get_speech_processor()
    return processor.text_to_speech_cloud(text, language_code)


def speech_to_text_local(audio_file_path: Union[str, Path]) -> str:
    """
    Convenience function for local speech-to-text.
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Transcribed text
    """
    processor = get_speech_processor()
    return processor.speech_to_text_local(audio_file_path)


def text_to_speech_local(text: str, lang: str = 'en') -> Tuple[str, bytes, str]:
    """
    Convenience function for local text-to-speech.
    
    Args:
        text: Text to convert to speech
        lang: Language code
        
    Returns:
        Tuple of (text, audio_bytes, temp_file_path)
    """
    processor = get_speech_processor()
    return processor.text_to_speech_local(text, lang)


if __name__ == "__main__":
    """
    Test examples demonstrating speech processing functionality.
    """
    print("Testing Speech Processing Module...")
    print("=" * 50)
    
    try:
        processor = get_speech_processor()
        
        # Display service info
        info = processor.get_service_info()
        print(f"Google Cloud Available: {info['google_cloud_available']}")
        print(f"Local STT Available: {info['local_stt_available']}")
        print(f"Local TTS Available: {info['local_tts_available']}")
        print(f"Speech Client Initialized: {info['speech_client_initialized']}")
        print(f"TTS Client Initialized: {info['tts_client_initialized']}")
        print(f"Available Languages: {info['available_languages']}")
        
        # Test local TTS (if available)
        if processor.local_tts_available:
            print(f"\nTesting Local Text-to-Speech:")
            print("-" * 35)
            
            test_text = "I have a fever and headache"
            try:
                text, audio_bytes, temp_path = processor.text_to_speech_local(test_text, 'en')
                print(f"Text: \"{text}\"")
                print(f"Audio size: {len(audio_bytes)} bytes")
                print(f"Temp file: {temp_path}")
                
                # Clean up temp file
                processor.cleanup_temp_file(temp_path)
                print("✓ Temp file cleaned up")
                
            except Exception as e:
                print(f"Local TTS test failed: {e}")
        else:
            print("⚠️ Local TTS not available - skipping test")
        
        # Test cloud services (if available)
        if processor.google_cloud_available:
            print(f"\nTesting Cloud Services:")
            print("-" * 25)
            print("Note: Cloud services require valid credentials and audio data")
            print("Skipping actual cloud tests in demo mode")
        else:
            print(f"\n⚠️ Google Cloud credentials not found")
            print("To test cloud services:")
            print("1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            print("2. Ensure you have valid Google Cloud Speech and TTS APIs enabled")
        
        # Show available languages
        print(f"\nAvailable Languages:")
        print("-" * 20)
        languages = processor.get_available_languages()
        for code, name in list(languages.items())[:10]:  # Show first 10
            print(f"{code}: {name}")
        print(f"... and {len(languages) - 10} more languages")
        
        print(f"\n✓ Speech processing module testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        raise
