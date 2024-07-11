from langchain.output_parsers import PydanticOutputParser
from langchain.tools import tool, Tool
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from data.validation import UserProfile

user_sub = [
    {"user_id": "1", "subscription": "premium"},
    {"user_id": "2", "subscription": "free"},
    {"user_id": "3", "subscription": "premium"},
]
user_info = [
    {
        "name": "Michael Jackson",
        "email": "michaeljackson@gmail.com",
        "user_id": "1",
        "phone": "0452 333 666",
        "language": "English",
    },
    {
        "name": "John Doe",
        "email": "john@doe.com",
        "user_id": "2",
        "phone": "0452 333 667",
        "language": "Spanish",
    },
    {
        "name": "Carl Sagan",
        "email": "carl@sagan.com",
        "user_id": "3",
        "phone": "0452 333 668",
        "language": "Italian",
    },
    {
        "name": "XYZ",
        "email": "xyz@gmail.com",
        "user_id": "4",
        "phone": "123456789",
        "language": "Italian",
    },
]
 

@tool("user_info_db", return_direct=True)
def search_user_info_on_db(email: str):
    """Searches users by email"""
    return list(filter(lambda user: user["email"] == email, user_info))


@tool("user_subscription_db", return_direct=True)
def search_user_subscription_on_db(id: str):
    """Searches users subscription by user id"""
    return list(filter(lambda user: user["user_id"] == id, user_sub))
