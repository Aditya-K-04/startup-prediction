from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import time
import sys
sys.path.append('C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/config')
from api_keys import YOUTUBE_API_KEY

class YouTubeDataCollector:
    def __init__(self, api_key=YOUTUBE_API_KEY):
        """Initialize YouTube API client"""
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    def search_channels(self, query, max_results=10):
        """Search for channels by query"""
        print(f"Searching for channels: {query}")
        
        try:
            request = self.youtube.search().list(
                part='snippet',
                q=query,
                type='channel',
                maxResults=max_results
            )
            response = request.execute()
            
            channels = []
            for item in response.get('items', []):
                channels.append({
                    'channel_id': item['id']['channelId'],
                    'channel_title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'query': query
                })
            
            print(f"Found {len(channels)} channels")
            return channels
            
        except HttpError as e:
            print(f"HTTP Error: {e}")
            return []
    
    def get_channel_statistics(self, channel_id):
        """Get detailed statistics for a channel"""
        try:
            request = self.youtube.channels().list(
                part='statistics,snippet,contentDetails',
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                return None
            
            item = response['items'][0]
            stats = item['statistics']
            snippet = item['snippet']
            
            return {
                'channel_id': channel_id,
                'channel_title': snippet['title'],
                'description': snippet.get('description', ''),
                'published_at': snippet['publishedAt'],
                'subscriber_count': int(stats.get('subscriberCount', 0)),
                'view_count': int(stats.get('viewCount', 0)),
                'video_count': int(stats.get('videoCount', 0)),
                'comment_count': int(stats.get('commentCount', 0))
            }
            
        except HttpError as e:
            print(f"Error fetching channel {channel_id}: {e}")
            return None
    
    def get_channel_videos(self, channel_id, max_results=10):
        """Get recent videos from a channel"""
        try:
            # Get uploads playlist ID
            request = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                return []
            
            uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Get videos from uploads playlist
            request = self.youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=max_results
            )
            response = request.execute()
            
            videos = []
            for item in response.get('items', []):
                videos.append({
                    'video_id': item['snippet']['resourceId']['videoId'],
                    'video_title': item['snippet']['title'],
                    'published_at': item['snippet']['publishedAt'],
                    'channel_id': channel_id
                })
            
            return videos
            
        except HttpError as e:
            print(f"Error fetching videos: {e}")
            return []
    
    def get_video_statistics(self, video_id):
        """Get statistics for a specific video"""
        try:
            request = self.youtube.videos().list(
                part='statistics,snippet,contentDetails',
                id=video_id
            )
            response = request.execute()
            
            if not response.get('items'):
                return None
            
            item = response['items'][0]
            stats = item['statistics']
            snippet = item['snippet']
            
            return {
                'video_id': video_id,
                'video_title': snippet['title'],
                'published_at': snippet['publishedAt'],
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'duration': item['contentDetails']['duration']
            }
            
        except HttpError as e:
            print(f"Error fetching video {video_id}: {e}")
            return None
    
    def collect_startup_channels_data(self, startup_queries, max_channels_per_query=5):
        """Collect channel data for multiple startup-related queries"""
        print("Starting YouTube data collection...")
        
        all_channels = []
        
        for query in startup_queries:
            channels = self.search_channels(query, max_results=max_channels_per_query)
            
            for channel in channels:
                # Get detailed statistics
                stats = self.get_channel_statistics(channel['channel_id'])
                if stats:
                    all_channels.append(stats)
                
                # Rate limiting
                time.sleep(1)
        
        df = pd.DataFrame(all_channels)
        print(f"Collected data for {len(df)} channels")
        
        return df
    
    def save_data(self, df, output_path):
        """Save collected data to CSV"""
        df.to_csv(output_path, index=False)
        print(f"Saved YouTube data to {output_path}")


# Usage example
if __name__ == "__main__":
    collector = YouTubeDataCollector()
    
    # Example startup-related queries
    startup_queries = [
        "tech startup",
        "kickstarter campaign",
        "crowdfunding project",
        "startup pitch",
        "product launch"
    ]
    
    df = collector.collect_startup_channels_data(startup_queries, max_channels_per_query=5)
    collector.save_data(df, 'C:/Users/chand/OneDrive/Desktop/7th sem/startup-prediction/data/raw/youtube_channels.csv')
