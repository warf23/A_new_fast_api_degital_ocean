from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import validators
from fastapi.middleware.cors import CORSMiddleware
import re
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

# FastAPI app instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Define a request body model
class SummarizeRequest(BaseModel):
  groq_api_key: str
  youtube_api_key: str
  url: str
  language: str = "English"

# Define the prompt template (unchanged)
prompt_template = """
Content Summary Request

Language: {language}
Word Count: Approximately 300 words
Source: {text}

Objective:
Provide a concise yet comprehensive summary of the given content in the specified language. The summary should be accessible to readers unfamiliar with the original material.

Key Focus Areas:
1. Main points and central themes
2. Key arguments and supporting evidence
3. Significant conclusions or findings
4. Notable insights or implications
5. Methodologies used (if applicable)

Summary Guidelines:
- Begin with a brief introduction contextualizing the content.
- Organize information logically, using clear transitions between ideas.
- Prioritize the most crucial information from the source material.
- Maintain objectivity, avoiding personal interpretations or biases.
- Include relevant statistics, data points, or examples that substantiate main ideas.
- Conclude with the overarching message or significance of the content.

Additional Considerations:
- Identify any limitations, potential biases, or areas of controversy in the source material.
- Highlight any unique or innovative aspects of the content.
- If relevant, briefly mention the credibility or expertise of the source.

Formatting:
- Use clear, concise language appropriate for the target audience.
- Employ bullet points or numbered lists for clarity when presenting multiple related points.
- Include subheadings if it enhances readability and organization.

Note: Ensure the summary stands alone as an informative piece, providing value even without access to the original content.
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])

# Language options
language_codes = {'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 
  'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'}

# Function to extract video ID from URL
def extract_video_id(url):
  video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
  if video_id_match:
      return video_id_match.group(1)
  return None

# Function to get video details using YouTube API
def get_video_details(api_key, video_id):
  youtube = build('youtube', 'v3', developerKey=api_key)
  request = youtube.videos().list(
      part="snippet",
      id=video_id
  )
  response = request.execute()
  return response['items'][0]['snippet']['title'], response['items'][0]['snippet']['description']

# Function to get transcript
def get_transcript(video_id, language_code):
  try:
      transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
      return " ".join([entry['text'] for entry in transcript])
  except Exception as e:
      print(f"An error occurred: {e}")
      return None

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
  groq_api_key = request.groq_api_key
  youtube_api_key = request.youtube_api_key
  url = request.url
  language = request.language

  # Validate input
  if not validators.url(url):
      raise HTTPException(status_code=400, detail="Invalid URL")

  if language not in language_codes:
      raise HTTPException(status_code=400, detail="Invalid language")

  try:
      # Initialize the language model
      model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

      # Extract video ID and get details
      video_id = extract_video_id(url)
      if not video_id:
          raise HTTPException(status_code=400, detail="Invalid YouTube URL")

      title, description = get_video_details(youtube_api_key, video_id)
      transcript = get_transcript(video_id, language_codes[language])

      if not transcript:
          raise HTTPException(status_code=404, detail="Transcript not available")

      # Combine video information
      combined_text = f"Title: {title}\n\nDescription: {description}\n\nTranscript: {transcript}"

      # Create the chain
      chain = (
          {"text": RunnablePassthrough(), "language": lambda _: language}
          | prompt
          | model
          | StrOutputParser()
      )

      # Run the chain
      output = chain.invoke(combined_text)

      return {"summary": output}

  except Exception as e:
      raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

# Run with Uvicorn
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)