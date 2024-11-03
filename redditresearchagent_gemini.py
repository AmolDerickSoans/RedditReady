import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from textblob import TextBlob
import nest_asyncio
import praw
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

@dataclass
class Config:
    """Configuration settings for the Reddit Research Agent"""
    monitoring_duration: int = 21600  # 6 hours in seconds
    check_interval: int = 3600  # 1 hour in seconds
    max_replies_per_thread: int = 4
    upvote_ratio_threshold: float = 0.05
    rate_limit_delay: int = 120  # seconds between API calls
    min_upvotes: int = 5

@dataclass
class ResearchData:
    """Structure for storing research data"""
    research_id: str
    original_prompt: str
    subreddit: str
    style_template: str
    post: Dict
    interactions: Dict
    
    def to_json(self) -> str:
        """Convert the research data to JSON string"""
        return json.dumps(asdict(self), indent=2)

class GeminiWrapper:
    """Wrapper for Gemini API interactions"""
    def __init__(self, temperature=0.6):
        self.model = genai.GenerativeModel(
            model_name='gemini-pro',
            generation_config={
                'temperature': temperature,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
        )
    
    async def generate_text(self, prompt: str) -> str:
        """Generate text using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating text with Gemini: {str(e)}")
            return ""

class RedditAPI:
    """Handler for Reddit API interactions"""
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            username=os.getenv('REDDIT_USERNAME'),
            password=os.getenv('REDDIT_PASSWORD')
        )
        self.last_api_call = 0
        
    def _respect_rate_limit(self):
        """Ensure we don't exceed Reddit's rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < Config.rate_limit_delay:
            time.sleep(Config.rate_limit_delay - time_since_last_call)
        self.last_api_call = time.time()

    def get_subreddit(self, subreddit_name: str) -> Optional[praw.models.Subreddit]:
        """Safely get a subreddit instance"""
        self._respect_rate_limit()
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            # Test if the subreddit is accessible
            subreddit.title
            return subreddit
        except Exception as e:
            print(f"Error accessing subreddit: {str(e)}")
            return None

class SubredditAnalysisAgent:
    """Agent for analyzing subreddit patterns and content style"""
    def __init__(self, gemini: GeminiWrapper):
        self.gemini = gemini
        
    async def analyze_subreddit(self, subreddit: praw.models.Subreddit) -> str:
        """Analyze subreddit and return posting style template"""
        # Collect recent popular posts
        posts = []
        for post in subreddit.hot(limit=10):
            posts.append({
                'title': post.title,
                'content': post.selftext if hasattr(post, 'selftext') else '',
                'score': post.score
            })
        
        # Generate style template
        prompt = """
        Analyze these recent posts and identify:
        1. Common writing styles
        2. Typical post structure
        3. Popular phrases and terminology
        4. Engagement patterns
        
        Posts:
        {}
        
        Provide a structured template for creating posts in this subreddit.
        """.format(json.dumps(posts, indent=2))
        
        return await self.gemini.generate_text(prompt)

class PromptGenerationAgent:
    """Agent for creating Reddit-optimized posts"""
    def __init__(self, gemini: GeminiWrapper):
        self.gemini = gemini
    
    async def generate_post(self, style_guide: str, research_prompt: str) -> str:
        """Generate a Reddit post based on style guide and research prompt"""
        prompt = """
        Create a Reddit post following this style guide:
        {}
        
        Research Topic:
        {}
        
        Generate a post that will encourage meaningful discussion and responses.
        """.format(style_guide, research_prompt)
        
        return await self.gemini.generate_text(prompt)

class ResponseAnalysisAgent:
    """Agent for analyzing and processing responses"""
    def __init__(self, gemini: GeminiWrapper):
        self.gemini = gemini
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob"""
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def should_reply(self, comment: praw.models.Comment, post_score: int) -> bool:
        """Determine if a comment warrants a reply"""
        return (
            comment.score > Config.min_upvotes and
            comment.score > (post_score * Config.upvote_ratio_threshold)
        )
    
    async def generate_reply(self, context: str) -> str:
        """Generate a reply based on context"""
        prompt = """
        Based on this conversation context:
        {}
        
        Generate a thoughtful and engaging reply that adds value to the discussion.
        """.format(context)
        
        return await self.gemini.generate_text(prompt)

class DataCollectionAgent:
    """Agent for collecting and storing research data"""
    def __init__(self):
        self.data: Dict[str, ResearchData] = {}
        
    def initialize_research(
        self,
        research_id: str,
        prompt: str,
        subreddit: str,
        style_template: str
    ) -> None:
        """Initialize a new research entry"""
        self.data[research_id] = ResearchData(
            research_id=research_id,
            original_prompt=prompt,
            subreddit=subreddit,
            style_template=style_template,
            post={},
            interactions={
                'replies': [],
                'metrics': {
                    'total_engagement': 0,
                    'sentiment_overview': {},
                    'key_insights': []
                }
            }
        )
    
    def update_post(self, research_id: str, post_data: Dict) -> None:
        """Update post information"""
        self.data[research_id].post = post_data
    
    def add_reply(self, research_id: str, reply_data: Dict) -> None:
        """Add a reply to the research data"""
        self.data[research_id].interactions['replies'].append(reply_data)
    
    def save_research(self, research_id: str, filepath: str) -> None:
        """Save research data to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.data[research_id].to_json())

class RedditResearchAgent:
    """Main agent coordinating the Reddit research process"""
    def __init__(self):
        self.gemini = GeminiWrapper(temperature=0.6)
        self.reddit_api = RedditAPI()
        self.subreddit_agent = SubredditAnalysisAgent(self.gemini)
        self.prompt_agent = PromptGenerationAgent(self.gemini)
        self.response_agent = ResponseAnalysisAgent(self.gemini)
        self.data_agent = DataCollectionAgent()
        
    async def run_research(
        self,
        research_prompt: str,
        subreddit_name: str,
        research_id: str = None
    ) -> Optional[str]:
        """Run the complete research process"""
        if research_id is None:
            research_id = f"research_{int(time.time())}"
            
        # Initialize subreddit
        subreddit = self.reddit_api.get_subreddit(subreddit_name)
        if not subreddit:
            print(f"Could not access subreddit: {subreddit_name}")
            return None
            
        # Analyze subreddit and generate post
        style_template = await self.subreddit_agent.analyze_subreddit(subreddit)
        post_content = await self.prompt_agent.generate_post(style_template, research_prompt)
        
        # Initialize research data
        self.data_agent.initialize_research(
            research_id,
            research_prompt,
            subreddit_name,
            style_template
        )
        
        # Make the post
        try:
            post = subreddit.submit(
                title=post_content.split('\n')[0],  # First line as title
                selftext='\n'.join(post_content.split('\n')[1:])  # Rest as content
            )
            
            self.data_agent.update_post({
                'id': post.id,
                'content': post_content,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'active'
            })
        except Exception as e:
            print(f"Error posting to Reddit: {str(e)}")
            return None
        
        # Monitor responses
        start_time = time.time()
        generated_replies = set()
        
        while time.time() - start_time < Config.monitoring_duration:
            try:
                post.refresh()
                
                # Check if post was removed
                if hasattr(post, 'removed'):
                    print("Post was removed by moderators")
                    self.data_agent.update_post({
                        'status': 'removed',
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    break
                
                # Process new comments
                post.comments.replace_more(limit=None)
                for comment in post.comments.list():
                    if comment.id not in generated_replies and \
                       self.response_agent.should_reply(comment, post.score):
                        # Analyze comment
                        sentiment = self.response_agent.analyze_sentiment(comment.body)
                        
                        # Store comment data
                        self.data_agent.add_reply({
                            'id': comment.id,
                            'content': comment.body,
                            'upvotes': comment.score,
                            'sentiment': sentiment,
                            'timestamp': datetime.fromtimestamp(comment.created_utc).isoformat(),
                            'is_bot_generated': False
                        })
                        
                        # Generate and post reply if needed
                        if len(generated_replies) < Config.max_replies_per_thread:
                            reply_content = await self.response_agent.generate_reply(
                                f"Original Post: {post_content}\n\nComment: {comment.body}"
                            )
                            
                            try:
                                reply = comment.reply(reply_content)
                                generated_replies.add(reply.id)
                                
                                # Store reply data
                                self.data_agent.add_reply({
                                    'id': reply.id,
                                    'content': reply_content,
                                    'upvotes': 0,
                                    'sentiment': self.response_agent.analyze_sentiment(reply_content),
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'is_bot_generated': True
                                })
                            except Exception as e:
                                print(f"Error posting reply: {str(e)}")
                
                # Wait before next check
                await asyncio.sleep(Config.check_interval)
                
            except Exception as e:
                print(f"Error monitoring responses: {str(e)}")
                await asyncio.sleep(Config.check_interval)
        
        # Save research data
        self.data_agent.save_research(
            research_id,
            f"research_data_{research_id}.json"
        )
        
        return research_id

# Usage example
async def main():
    agent = RedditResearchAgent()
    research_prompt = "What features do users want in next-generation smartphones?"
    subreddit_name = "smartphones"
    
    research_id = await agent.run_research(research_prompt, subreddit_name)
    if research_id:
        print(f"Research completed successfully. ID: {research_id}")
    else:
        print("Research failed to complete")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())