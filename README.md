# Reddit Research Agent 🤖

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)



                                                                   
<pre>
  ▗▄▄▖ ▗▄▄▄▖▗▄▄▄  ▗▄▄▄  ▗▄▄▄▖▗▄▄▄▖    ▗▄▄▖ ▗▄▄▄▖ ▗▄▖ ▗▄▄▄ ▗▖  ▗▖
  ▐▌ ▐▌▐▌   ▐▌  █ ▐▌  █   █    █      ▐▌ ▐▌▐▌   ▐▌ ▐▌▐▌  █ ▝▚▞▘ 
  ▐▛▀▚▖▐▛▀▀▘▐▌  █ ▐▌  █   █    █      ▐▛▀▚▖▐▛▀▀▘▐▛▀▜▌▐▌  █  ▐▌  
  ▐▌ ▐▌▐▙▄▄▖▐▙▄▄▀ ▐▙▄▄▀ ▗▄█▄▖  █      ▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌▐▙▄▄▀  ▐▌  
</pre>
                                                                
                                                                
                                                               

                                                     

An intelligent agent system that conducts automated research on Reddit by creating posts, analyzing responses, and engaging in meaningful discussions using Google's Gemini AI. This tool helps researchers, marketers, and analysts gather authentic community feedback and insights through natural interactions.

## 🌟 Features

- **Automated Subreddit Analysis**: Analyzes subreddit patterns, writing styles, and engagement metrics to optimize post creation
- **AI-Powered Content Generation**: Uses Google's Gemini AI to generate contextually relevant posts and responses
- **Smart Engagement**: Automatically identifies and responds to high-value comments based on configurable metrics
- **Sentiment Analysis**: Tracks and analyzes the emotional tone of conversations
- **Data Collection**: Comprehensive storage of research data in structured JSON format
- **Rate Limiting**: Built-in safeguards to respect Reddit's API limitations
- **Configurable Parameters**: Easily adjustable settings for research duration, engagement thresholds, and more

## 📋 Prerequisites

- Python 3.8 or higher
- Reddit API credentials
- Google Gemini API key
- Required Python packages (see Installation)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reddit-research-agent.git
cd reddit-research-agent
```

2. Install required packages:
```bash
pip install praw google-generativeai textblob nest-asyncio python-dotenv
```

3. Set up environment variables in `.env`:
```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password
GOOGLE_API_KEY=your_gemini_api_key
```

## 🚀 Usage

### Basic Usage

```python
from reddit_research_agent import RedditResearchAgent
import asyncio

async def main():
    agent = RedditResearchAgent()
    
    # Start a research session
    research_id = await agent.run_research(
        research_prompt="What features do users want in next-generation smartphones?",
        subreddit_name="smartphones"
    )
    
    if research_id:
        print(f"Research completed successfully. ID: {research_id}")
    else:
        print("Research failed to complete")

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration

Customize the agent's behavior by modifying the `Config` class parameters:

```python
@dataclass
class Config:
    monitoring_duration: int = 21600  # 6 hours
    check_interval: int = 3600        # 1 hour
    max_replies_per_thread: int = 4
    upvote_ratio_threshold: float = 0.05
    rate_limit_delay: int = 120
    min_upvotes: int = 5
```

## 📊 Data Collection

The agent stores research data in JSON format with the following structure:

```json
{
  "research_id": "research_1234567890",
  "original_prompt": "Research question/prompt",
  "subreddit": "subreddit_name",
  "style_template": "Generated posting style guide",
  "post": {
    "id": "post_id",
    "content": "Post content",
    "timestamp": "ISO timestamp",
    "status": "active/removed"
  },
  "interactions": {
    "replies": [
      {
        "id": "comment_id",
        "content": "Comment content",
        "upvotes": 10,
        "sentiment": {
          "polarity": 0.5,
          "subjectivity": 0.5
        },
        "timestamp": "ISO timestamp",
        "is_bot_generated": false
      }
    ],
    "metrics": {
      "total_engagement": 0,
      "sentiment_overview": {},
      "key_insights": []
    }
  }
}
```

## 🏗️ Architecture

The system consists of several specialized agents working together:

1. **SubredditAnalysisAgent**: Analyzes subreddit patterns and content style
2. **PromptGenerationAgent**: Creates Reddit-optimized posts
3. **ResponseAnalysisAgent**: Processes and analyzes responses
4. **DataCollectionAgent**: Manages research data storage
5. **GeminiWrapper**: Handles AI text generation
6. **RedditAPI**: Manages Reddit interactions

## ⚠️ Important Notes

- Ensure compliance with Reddit's [API Terms of Service](https://www.reddit.com/dev/api-terms)
- Monitor and adjust rate limiting settings to avoid API restrictions
- Be transparent about automated interactions when required by subreddit rules
- Regularly backup research data
- Review generated content before deployment in production environments

## 🔍 Advanced Features

### Custom Style Templates

You can provide custom style templates for specific subreddits:

```python
custom_template = """
Title Format: [Discussion] Clear, engaging question
Body Format:
- Introduction (2-3 sentences)
- Main question/topic
- Additional context
- Call to action
"""

agent = RedditResearchAgent()
await agent.subreddit_agent.analyze_subreddit(
    subreddit,
    style_template=custom_template
)
```

### Sentiment Analysis Customization

Modify sentiment analysis thresholds:

```python
from textblob import TextBlob

class CustomResponseAnalysisAgent(ResponseAnalysisAgent):
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        analysis = TextBlob(text)
        # Custom sentiment thresholds
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity,
            'is_positive': analysis.sentiment.polarity > 0.2
        }
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PRAW](https://praw.readthedocs.io/) - Python Reddit API Wrapper
- [Google Gemini AI](https://ai.google.dev/) - AI text generation
- [TextBlob](https://textblob.readthedocs.io/) - Text processing and sentiment analysis

## 📧 Contact

Your Name - [@yourusername](https://twitter.com/yourusername) - email@example.com

Project Link: [https://github.com/yourusername/reddit-research-agent](https://github.com/yourusername/reddit-research-agent)
