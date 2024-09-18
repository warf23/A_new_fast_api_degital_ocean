from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_extraction_chain 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import validators
from fastapi.middleware.cors import CORSMiddleware

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
    url: str
    language: str = "English"

# Define the prompt template
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

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    groq_api_key = request.groq_api_key
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

        # Load the URL content
        if "youtube.com" in url:
            loader = YoutubeLoader.from_youtube_url(url, language=language_codes[language], add_video_info=True)
        else:
            loader = UnstructuredURLLoader(
                urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0"}
            )

        docs = loader.load()

        # Combine the documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        combined_text = " ".join([doc.page_content for doc in texts])

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


# Update the request model
class AdvancedSummarizeRequest(BaseModel):
    groq_api_key: str
    url: str
    language: str = "English"
    length: str = "Short"
    expertise_level: str = "intermediate"
    focus_area: str = "general"
    include_formulas: bool = False

# Update the prompt template
advanced_prompt_template = """
Content Summary Request

Language: {language}
Word Count: {length} words
Source: {text}
Expertise Level: {expertise_level}
Focus Area: {focus_area}
Include Formulas: {include_formulas}

Objective:
Provide a comprehensive summary of the given content in the specified language, tailored to the indicated expertise level and focus area. The summary should be approximately {length} words and accessible to readers familiar with the {expertise_level} level of {focus_area}.

Key Focus Areas:
1. Main points and central themes related to {focus_area}
2. Key arguments and supporting evidence
3. Significant conclusions or findings
4. Notable insights or implications for {focus_area}
5. Methodologies or techniques used (if applicable)

Summary Guidelines:
- Begin with a brief introduction contextualizing the content within {focus_area}.
- Organize information logically, using clear transitions between ideas.
- Prioritize information most relevant to {focus_area} and {expertise_level}.
- Maintain objectivity, avoiding personal interpretations or biases.
- Include relevant statistics, data points, or examples that substantiate main ideas.
- Conclude with the overarching message or significance of the content for {focus_area}.

Focus Area-Specific Guidelines:
- Emphasize domain-specific concepts, methodologies, or frameworks relevant to {focus_area}.
- Use terminology appropriate for the {expertise_level}, providing brief explanations if necessary.
- If {include_formulas} is True and the content contains mathematical or scientific formulas:
  * Include and highlight key formulas or equations.
  * Explain the significance and application of these formulas within the context.
  * Ensure formulas are clearly formatted and easy to read.

Formatting:
- Use clear, concise language appropriate for the {expertise_level} in {focus_area}.
- Employ bullet points or numbered lists for clarity when presenting multiple related points.
- Include subheadings if it enhances readability and organization.

Note: Ensure the summary stands alone as an informative piece, providing value even without access to the original content, while being specifically tailored to {focus_area} at an {expertise_level} level.
"""

advanced_prompt = PromptTemplate(
    template=advanced_prompt_template, 
    input_variables=["text", "language", "length", "expertise_level", "focus_area", "include_formulas"]
)

@app.post("/summarize/advanced")
async def summarize_advanced(request: AdvancedSummarizeRequest):
    groq_api_key = request.groq_api_key
    url = request.url
    language = request.language
    length = request.length
    expertise_level = request.expertise_level
    focus_area = request.focus_area
    include_formulas = request.include_formulas

    # Validate input
    if not validators.url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    if language not in language_codes:
        raise HTTPException(status_code=400, detail="Invalid language")

    try:
        # Initialize the language model
        model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

        # Load the URL content (same as in the original summarize function)
        if "youtube.com" in url:
            loader = YoutubeLoader.from_youtube_url(url, language=language_codes[language], add_video_info=True)
        else:
            loader = UnstructuredURLLoader(
                urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0"}
            )

        docs = loader.load()

        # Combine the documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        combined_text = " ".join([doc.page_content for doc in texts])

        # Create the chain
        chain = (
            {
                "text": RunnablePassthrough(), 
                "language": lambda _: language,
                "length": lambda _: length,
                "expertise_level": lambda _: expertise_level,
                "focus_area": lambda _: focus_area,
                "include_formulas": lambda _: str(include_formulas)
            }
            | advanced_prompt
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