import os
from pydub import AudioSegment

def separate_audio(input_file, output_path, min_length=1.0):
    """
    Splits an audio file into segments of at least `min_length` seconds.

    Args:
        input_file (str): Path to the input audio file.
        output_path (str): Directory to save the output audio segments.
        min_length (float): Minimum length of each segment in seconds.
    """
    audio = AudioSegment.from_file(input_file)
    min_length_ms = int(min_length * 1000)  # pydub works with milliseconds
    total_length = len(audio)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    file_names = []

    os.makedirs(output_path, exist_ok=True)

    segment_count = 0
    for start_time in range(0, total_length, min_length_ms):
        end_time = min(start_time + min_length_ms, total_length)
        segment = audio[start_time:end_time]
        if len(segment) < min_length_ms:
            break  # Avoid final segment if it's too short

        output_file = os.path.join(output_path, f"{base_name}_part{segment_count + 1}.wav")
        segment.export(output_file, format="wav")
        file_names += [output_file]
        segment_count += 1

    print(f"Exported {segment_count} segments from {input_file}.")
    return file_names


separate_audio("data/target/desert_eagle/original/desert_eagle.wav", "data/target/desert_eagle")