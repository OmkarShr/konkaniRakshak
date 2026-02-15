
import asyncio
import io
import soundfile as sf
import base64
import aiohttp
import os
import sys

# We need to run this where Parler TTS is installed (pipeline-english container)

async def main():
    print("Loading TTS...")
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import torch

    device = "cpu"
    
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    
    prompt = "Hello world. This is a test."
    description = "Sanjay speaks clearly."
    
    print(f"Generating audio for: '{prompt}'")
    
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    
    # Resample to 16000 (Parler is 44100 usually, but let's check config, actually we just save and resample or send as is if STT handles it. STT expects 16k usually)
    # NeMo STT usually expects 16k.
    import librosa
    import numpy as np
    
    # We might not have librosa. Parler output is 44.1kHz typically.
    # Simple resampling if no librosa:
    # Actually, stt_service.py takes 'sample_rate' param.
    # But usually models are trained on 16k.
    # Let's try to resample using scipy if available.
    
    target_sr = 16000
    orig_sr = model.config.sampling_rate
    
    import scipy.signal
    num_samples = int(len(audio_arr) * target_sr / orig_sr)
    audio_resampled = scipy.signal.resample(audio_arr, num_samples)
    
    # Convert to int16
    audio_int16 = (audio_resampled * 32767).astype(np.int16)
    
    audio_bytes = audio_int16.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    url = "http://multilingual-stt:50052/transcribe"
    payload = {
        "audio": audio_b64,
        "sample_rate": 16000,
        "language": "hi"
    }
    
    print(f"Sending to STT (lang='hi')...")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            print(f"Status: {resp.status}")
            txt = await resp.text()
            print(f"Response: {txt}")

if __name__ == "__main__":
    asyncio.run(main())
