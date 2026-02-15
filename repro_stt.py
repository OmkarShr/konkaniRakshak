
import aiohttp
import asyncio
import base64
import wave
import io

async def test_stt_lang(lang_code):
    # Create a small silent wav
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b'\x00' * 32000) # 1 sec silence
    
    wav_bytes = buffer.getvalue()
    audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
    
    url = "http://localhost:50052/transcribe"
    payload = {
        "audio": audio_b64,
        "sample_rate": 16000,
        "language": lang_code
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                print(f"Code: {lang_code} -> Status: {resp.status}")
                if resp.status != 200:
                    text = await resp.text()
                    print(f"  Error: {text}")
                else:
                    print("  Success")
    except Exception as e:
        print(f"Code: {lang_code} -> Exception: {e}")

async def main():
    codes = ["en", "english", "eng", "en_IN", "en-IN", "English", "ur"]
    requests = [test_stt_lang(c) for c in codes]
    await asyncio.gather(*requests)

if __name__ == "__main__":
    asyncio.run(main())
