import whisper
from langchain.chains import LLMChain
# from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import SimpleMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
import librosa
from data.validation import PhoneCallTicket


def call_customer(query: str):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    model = whisper.load_model("base")
    # result = model.transcribe("assets/audio/customer_support.wav")
    audio, sr = librosa.load("assets/audio/customer_support.wav", sr=None)
    
    # Transcribe using whisper
    result = model.transcribe(audio, fp16=False)
    parser = PydanticOutputParser(pydantic_object=PhoneCallTicket)
    summary_prompt_template = """Write a concise summary of the following:

{text}
    
CONCISE SUMMARY IN ENGLISH:"""

    prefix_create_ticket = "You read Customer Call transcriptions and their summary and use the below output format instructions to answer:\n\n"
    suffix_create_ticket = """
{format_instructions}
Call Summary:
{call_summary}
Answer:
"""

    create_ticket_template = prefix_create_ticket + suffix_create_ticket

    summary_prompt = PromptTemplate(
        template=summary_prompt_template, input_variables=["text"]
    )
    ticket_prompt = PromptTemplate(
        template=create_ticket_template, input_variables=["call_summary"]
    )

    summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="call_summary")

    ticket_chain = LLMChain(llm=llm, prompt=ticket_prompt, output_key="ticket")

    sequential = SequentialChain(
        chains=[summary_chain, ticket_chain],
        input_variables=["text"],
        output_variables=["call_summary", "ticket"],
        memory=SimpleMemory(
            memories={"format_instructions": parser.get_format_instructions()}
        ),
        verbose=True,
    )

    completion = sequential(
        {
            "text": result["text"],
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return completion["ticket"]
 