import os
import subprocess
import random
import re
from typing import Optional, List


def get_available_formats(url: str) -> List[str]:
    """
    Get available format IDs for a YouTube video.
    """
    try:
        cmd = ["yt-dlp", "--list-formats", url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Parse the output to extract format IDs
            format_ids = []
            lines = result.stdout.split('\n')
            
            for line in lines:
                # Look for lines that start with a number (format ID)
                match = re.match(r'^(\d+)\s+', line)
                if match:
                    format_id = match.group(1)
                    # Skip audio-only formats (233, 234)
                    if format_id not in ['233', '234']:
                        format_ids.append(format_id)
            
            return format_ids
        else:
            print("Error listing formats:")
            print(result.stderr)
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def download_videos(dataset_data, output_dir="videos"):
    """
    Download videos from the dataset with improved format handling.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_id = dataset_data['video_id']
    url = dataset_data['url']
    start_time = dataset_data.get('start time', 0)
    end_time = dataset_data.get('end time', 30)
    
    output_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    if os.path.exists(output_path):
        return output_path
    
    # First, get available formats for this video
    print(f"Getting available formats for {video_id}...")
    available_formats = get_available_formats(url)
    
    if not available_formats:
        print(f"No available formats found for {video_id}")
        return None
    
    print(f"Available formats: {available_formats}")
    
    # Try formats in order of preference (lower resolution first for smaller files)
    # Format IDs: 269 (256x144), 230 (640x360), 605 (640x360), 232 (1280x720)
    preferred_order = ['269', '230', '605', '232']
    
    # Filter to only use available formats
    formats_to_try = [f for f in preferred_order if f in available_formats]
    
    # If none of our preferred formats are available, try any available format
    if not formats_to_try:
        formats_to_try = available_formats
    
    for format_id in formats_to_try:
        try:
            cmd = [
                "yt-dlp", "--format", format_id, 
                "--output", output_path, "--quiet",
                "--external-downloader", "ffmpeg",
                "--external-downloader-args", 
                f"ffmpeg:-ss {start_time} -t {end_time - start_time}",
                url
            ]

            print(f"Trying format ID: {format_id}")
            
            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if result.returncode == 0 and os.path.exists(output_path):
                print(f"Downloaded: {video_id} with format {format_id}")
                return output_path
            else:
                # If this format failed, try the next one
                print(f"Format {format_id} failed, trying next...")
                continue
                
        except Exception as e:
            print(f"Error with format {format_id}: {e}")
            continue
    
    # If all formats failed, try without external downloader
    for format_id in formats_to_try:
        try:
            print(f"Trying format {format_id} without external downloader...")
            cmd = [
                "yt-dlp", "--format", format_id, 
                "--output", output_path, "--quiet",
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(output_path):
                print(f"Downloaded: {video_id} with format {format_id} without external downloader")
                return output_path
        except Exception as e:
            print(f"Error with format {format_id} without external downloader: {e}")
            continue
    
    print(f"Failed to download: {video_id}")
    return None


def list_available_formats(url: str) -> None:
    """
    List all available formats for a YouTube video.
    """
    try:
        cmd = ["yt-dlp", "--list-formats", url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("Available formats:")
            print(result.stdout)
        else:
            print("Error listing formats:")
            print(result.stderr)
    except Exception as e:
        print(f"Error: {e}")


def download_with_custom_format(url: str, format_id: str, output_path: str, 
                               start_time: float = 0, duration: float = None) -> Optional[str]:
    """
    Download video with a specific format ID.
    """
    try:
        cmd = ["yt-dlp", "--format", format_id, "--output", output_path, "--quiet"]
        
        if duration:
            cmd.extend([
                "--external-downloader", "ffmpeg",
                "--external-downloader-args", 
                f"ffmpeg:-ss {start_time} -t {duration}"
            ])
        
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"Downloaded successfully: {output_path}")
            return output_path
        else:
            print(f"Download failed: {result.stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None
