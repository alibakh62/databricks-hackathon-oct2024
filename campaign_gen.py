import os
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_databricks import ChatDatabricks
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import chain as as_runnable
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
import json

max_num_turns = 3
from langgraph.pregel import RetryPolicy
import asyncio
from dotenv import load_dotenv

load_dotenv()


def add_messages(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right


def update_references(references, new_references):
    if not references:
        references = {}
    references.update(new_references)
    return references


def update_editor(editor, new_editor):
    if not editor:
        return new_editor
    return editor


class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the research analyis.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in (self.subsections if self.subsections else [])
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the research analyis")
    sections: List[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the research analyis.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in (self.sections if self.sections else []))
        return f"# {self.page_title}\n\n{sections}".strip()


class Queries(BaseModel):
    queries: List[str] = Field(
        description="Comprehensive list of search engine queries to answer the user's questions.",
    )


class Editor(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the editor.",
    )
    name: str = Field(
        description="One word First name of the editor without any Honorifics or titles."
    )
    role: str = Field(description="Role of the editor in the context of the topic.")
    description: str = Field(
        description="Description of the editor's focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="Comprehensive list of editors with their roles and affiliations.",
        # Add a pydantic validation/restriction to be at most M editors
    )


class InterviewState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    references: Annotated[Optional[dict], update_references]
    editor: Annotated[Optional[Editor], update_editor]


class AnswerWithCitations(BaseModel):
    answer: str = Field(
        description="Comprehensive answer to the user's question with citations.",
    )
    cited_urls: List[str] = Field(
        description="List of urls cited in the answer.",
    )

    @property
    def as_str(self) -> str:
        return f"{self.answer}\n\nCitations:\n\n" + "\n".join(
            f"[{i+1}]: {url}" for i, url in enumerate(self.cited_urls)
        )


def run_research_agent(
    example_topic,
    user_input_campaign_description,
    industry,
    refinement_request: Optional[str] = None,
):
    ki = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(model="gpt-5", api_key=ki)

    topic_instruction = example_topic
    if refinement_request:
        topic_instruction = (
            f"{example_topic}\n\nRefinement guidance: {refinement_request}"
        )

    # Generate outline
    direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Sr.Product Research Manager working at {industry} industry in the US. 
                Write an outline for a research analysis and sentiment analysis document for given product/topic. 
                Be comprehensive and specific.
                
                The output should be structured as follows:
                {{
                    "page_title": "Title of the analysis",
                    "sections": [
                        {{
                            "section_title": "Section Title",
                            "description": "Section description",
                            "subsections": [
                                {{
                                    "subsection_title": "Subsection Title",
                                    "description": "Subsection description"
                                }}
                            ]
                        }}
                    ]
                }}
                """,
            ),
            ("user", "{topic}"),
        ]
    )

    generate_outline_direct = direct_gen_outline_prompt | model.with_structured_output(
        Outline,
        include_raw=True  # Add this to help with debugging
    )

    try:
        initial_outline = generate_outline_direct.invoke(
            {"topic": topic_instruction, "industry": industry}
        )
        if isinstance(initial_outline, dict) and "parsed" in initial_outline:
            initial_outline = initial_outline["parsed"]
    except Exception as e:
        print(f"Error generating initial outline: {str(e)}")
        # Provide a basic fallback outline
        initial_outline = Outline(
            page_title=f"Research Analysis: {example_topic}",
            sections=[
                Section(
                    section_title="Overview",
                    description="General overview of the product/topic",
                    subsections=[
                        Subsection(
                            subsection_title="Introduction",
                            description="Brief introduction to the topic"
                        )
                    ]
                )
            ]
        )

    # print(initial_outline.as_str)

    # Generate perspectives
    gen_perspectives_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You need to select a diverse (and distinct) group of research analyis and sentiment analysis document Strategists who will work together to create a research analyis and sentiment analysis document on the topic. Each of them represents a different perspective, role, or affiliation related to this topic.\
        You can use other research analyis and sentiment analysis documents/ examples pages of related topics for inspiration. For each editor, add a description of what they will focus on.

        For each of topics in research analyis and sentiment analysis documents outline find some experts.
        {examples}""",
            ),
            ("user", "Topic of interest: {topic}"),
        ]
    ).partial(examples=initial_outline)

    gen_perspectives_chain = gen_perspectives_prompt | model.with_structured_output(
        Perspectives
    )

    perspectives = gen_perspectives_chain.invoke(topic_instruction)

    # perspectives.dict()

    def tag_with_name(ai_message: AIMessage, name: str):
        ai_message.name = name
        return ai_message

    def swap_roles(state: InterviewState, name: str):
        converted = []
        for message in state["messages"]:
            if isinstance(message, AIMessage) and message.name != name:
                message = HumanMessage(**message.dict(exclude={"type"}))
            converted.append(message)
        return {"messages": converted}

    def generate_question(state: InterviewState):
        gen_qn_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a product analyst tasked with creating a detailed research and analysis document for a new product launch. You are speaking with an industry expert to gather insights on market trends, consumer behavior, and competitor analysis. Ask focused and insightful questions, one at a time, to collect valuable information for your analysis.
        Be comprehensive and curious, gaining as much unique insight from the expert as possible to inform your research.

        Stay true to your specific perspective:

        {persona}""",
                ),
                MessagesPlaceholder(variable_name="messages", optional=True),
            ]
        )
        editor = state["editor"]
        gn_chain = (
            RunnableLambda(swap_roles).bind(name=editor.name)
            | gen_qn_prompt.partial(persona=editor.persona)
            | model
            | RunnableLambda(tag_with_name).bind(name=editor.name)
        )
        result = gn_chain.invoke(state)
        print("Persona", editor)
        print("Question:", result)
        return {"messages": [result]}

    gen_queries_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant. Query the search engine to answer the user's questions.",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )
    gen_queries_chain = gen_queries_prompt | model.with_structured_output(
        Queries, include_raw=True
    )

    gen_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a product analysis expert with extensive knowledge of product research. You are working with a product analyst who is developing a comprehensive research document for a product campaign. Use your expertise to provide actionable insights and data-driven recommendations that can inform the product analysis. Support your responses with credible sources such as market research reports or industry publications, and format citations as footnotes to help the analyst verify and incorporate the information into their document""",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )

    gen_answer_chain = gen_answer_prompt | model.with_structured_output(
        AnswerWithCitations, include_raw=True
    ).with_config(run_name="GenerateAnswer")

    # DDG
    # search_engine = DuckDuckGoSearchAPIWrapper()

    @tool
    def search_engine(query: str):
        """Search engine to the internet."""
        # print("In search engine")
        # print("query:", query)
        results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)
        return [{"content": r["body"], "url": r["href"]} for r in results]

    def gen_answer(
        state: InterviewState,
        config: Optional[RunnableConfig] = None,
        name: str = "expert",
        max_str_len: int = 15000,
    ):
        swapped_state = swap_roles(state, name)
        queries = gen_queries_chain.invoke(swapped_state)
        query_results = search_engine.batch(
            queries["parsed"].queries, config, return_exceptions=True
        )
        successful_results = [
            res for res in query_results if not isinstance(res, Exception)
        ]
        all_query_results = {
            res["url"]: res["content"]
            for results in successful_results
            for res in results
        }

        dumped = json.dumps(all_query_results)[:max_str_len]
        ai_message: AIMessage = queries["raw"]
        tool_call = queries["raw"].tool_calls[0]
        tool_id = tool_call["id"]
        tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
        swapped_state["messages"].extend([ai_message, tool_message])

        generated = gen_answer_chain.invoke(swapped_state)
        # print("Answer:", generated)
        cited_urls = set(generated["parsed"].cited_urls)

        cited_references = {
            k: v for k, v in all_query_results.items() if k in cited_urls
        }
        formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
        return {"messages": [formatted_message], "references": cited_references}

    max_num_turns = 3

    def route_messages(state: InterviewState, name: str = "expert"):
        messages = state["messages"]
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        if num_responses >= max_num_turns:
            return END
        last_question = messages[-2]
        if last_question.content.endswith("Thank you so much for your help!"):
            return END
        return "ask_question"

    builder = StateGraph(InterviewState)

    builder.add_node(
        "ask_question", generate_question, retry=RetryPolicy(max_attempts=5)
    )
    builder.add_node("answer_question", gen_answer, retry=RetryPolicy(max_attempts=5))
    builder.add_conditional_edges("answer_question", route_messages)
    builder.add_edge("ask_question", "answer_question")

    builder.add_edge(START, "ask_question")
    interview_graph = builder.compile(checkpointer=False).with_config(
        run_name="Conduct Interviews"
    )

    initial_states = [
        {
            "editor": editor,
            "messages": [
                AIMessage(
                    content=f"So you said you were doing a product research and sentiment analysis on {example_topic}?",
                    name="expert",
                )
            ],
        }
        for editor in perspectives.editors
    ]

    # Define a helper function to process a small batch of states
    def process_batch(batch):
        return interview_graph.batch(batch)

    # Define batch size and delay between batches
    batch_size = 5
    delay_between_batches = 2  # seconds

    # Process the initial_states in smaller batches
    # initial_states = [...]  # Your list of initial states
    batches = [
        initial_states[i : i + batch_size]
        for i in range(0, len(initial_states), batch_size)
    ]
    import time

    interview_results = []
    for batch in batches:
        results = process_batch(batch)
        interview_results.extend(results)
        time.sleep(delay_between_batches)  # Wait before processing the next batch
        # asyncio.sleep(delay_between_batches)  # Wait before starting the next batch

    def format_conversation(interview_state):
        messages = interview_state["messages"]
        convo = "\n".join(f"{m.name}: {m.content}" for m in messages)
        return f'Conversation with {interview_state["editor"].name}\n\n' + convo

    convos = "\n\n".join(
        [format_conversation(interview_state) for interview_state in interview_results]
    )

    # convos

    refine_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You’re a product analyst tasked with developing a detailed product and research analysis document for a new product launch. Based on your market research and expert interviews, create a comprehensive and specific analysis that covers market trends, competitive landscape, and consumer insights. Ensure the document is clear, concise, and tailored to the product's strategic goals, making it easy to follow while providing actionable insights for stakeholders Make it concise and easy to follow. \
    Topic you are writing about: {topic}
    You are being given a outline with topics and descriptions, fill those topics with information from the interviews and conversations.
    Outline:
    {old_outline}""",
            ),
            (
                "user",
                "Create a Product and Research Analysis Document based on your conversations with subject-matter experts:\n\n Base the report completely on conversations.\n\nConversations:\n\n{conversations}\n\nWrite the product and research analysis document:",
            ),
        ]
    )

    # Using turbo preview since the context can get quite long
    refine_outline_chain = refine_outline_prompt | model.with_structured_output(Outline)

    refined_outline = refine_outline_chain.invoke(
        {
            "topic": topic_instruction,
            "old_outline": initial_outline.as_str,
            "conversations": convos,
        }
    )

    # print(refined_outline.as_str)

    campaign_info = refined_outline.as_str

    example_emails = """1. Promotion of Deals and Discounts
    Subject Line: “Get $1 Coffee for Your Morning Buzz!”
    Content: A visually appealing email featuring images of fresh coffee or snacks with bold text highlighting the deal. They may include a coupon code or mention that the deal is available through the 7-Eleven app or in-store.
    CTA (Call-to-Action): "Redeem Now in the App" or "Find a Store Near You"
    2. Exclusive Offers for 7Rewards Members
    Subject Line: “Exclusive for You: Free Slurpee with Every Purchase!”
    Content: A personalized email addressing the recipient by name, showcasing exclusive offers for loyalty program members. It will often include details on how to earn points with each purchase and use them for rewards like free products.
    CTA: "Download the 7Rewards App" or "Claim Your Free Slurpee"
    3. New Product Launches
    Subject Line: “Try Our New Pumpkin Spice Latte Today!”
    Content: An email promoting a new seasonal product, such as a limited-edition flavor or item. It might include vibrant product imagery, a short description of the new item, and a time-limited offer.
    CTA: "Order Now" or "Try It at Your Local 7-Eleven"
    4. Seasonal or Holiday-Themed Promotions
    Subject Line: “Spooky Savings for Halloween: Buy One, Get One Free Snacks!”
    Content: A themed email with festive visuals related to the holiday. The body might include several offers, such as discounts on candy or holiday-themed snacks. Often, there are countdowns to create urgency.
    CTA: "Shop Now" or "Get Your Halloween Treats"
    5. Partnership and Co-branded Campaigns
    Subject Line: “Game Night Essentials: Get 2-for-1 Red Bull with Your Pizza!”
    Content: An email that promotes products from partner brands or popular gaming nights, offering special combo deals. There might be a tie-in with pop culture, movies, or other events.
    CTA: "Fuel Up for Game Night" or "Order Snacks and Drinks"
    6. Subscription Reminders
    Subject Line: “Never Run Out of Coffee Again: Sign Up for Our Coffee Subscription!”
    Content: An email promoting the 7-Eleven subscription service (for items like coffee or other essentials). It often includes a clear value proposition, such as saving money by subscribing.
    CTA: "Subscribe Now"
    """

    write_email_prompt = """
    You are a professional, creative email content writer for marketing emails.\n
        You will be given Audience Info,Campaign Info and Campaign topic (product info/ ) to write an email campaign. \n
        Campaign/Product Info researched by jr. researcher: {campaign_info} \n
        Campaign Topic: {example_topic} \n
        Additional refinement guidance from the marketer (may be blank): {refinement_request}
        Make sure the campaign is relevant to the audience and is a good fit for the product.
        Campaign description by Sr. Marketing Manager: {user_input_campaign_description}.
        You should not take critical decisions like giving out promotions based on research by jr. researcher, you should only take such decisions if instructed by  Sr. Marketing Manager.
        Now I will give you some example emails for marketing
        {example_emails}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", write_email_prompt),
            MessagesPlaceholder(variable_name="formatted_chat_history"),
            ("user", "{question}"),
        ]
    ).partial(
        campaign_info=campaign_info,
        example_topic=topic_instruction,
        user_input_campaign_description=user_input_campaign_description,
        example_emails=example_emails,
        refinement_request=refinement_request or "",
    )

    def extract_chat_history(chat_messages_array):
        return chat_messages_array[:-1]

    def format_chat_history_for_prompt(chat_messages_array):
        history = extract_chat_history(chat_messages_array)
        formatted_chat_history = []
        if len(history) > 0:
            for chat_message in history:
                if chat_message["role"] == "user":
                    formatted_chat_history.append(
                        HumanMessage(content=chat_message["content"])
                    )
                elif chat_message["role"] == "assistant":
                    formatted_chat_history.append(
                        AIMessage(content=chat_message["content"])
                    )
        return formatted_chat_history

    def extract_user_query_string(chat_messages_array):
        return chat_messages_array[-1]["content"]

    def format_context(docs):
        chunk_contents = [f"{d.page_content}\n" for d in docs]
        return "".join(chunk_contents)

    chain = (
        {
            "question": itemgetter("messages")
            | RunnableLambda(extract_user_query_string),
            "chat_history": itemgetter("messages")
            | RunnableLambda(extract_chat_history),
            "formatted_chat_history": itemgetter("messages")
            | RunnableLambda(format_chat_history_for_prompt),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    generated_email = chain.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a marketing email for the campaign mentioned ",
                }
            ]
        }
    )

    return campaign_info, generated_email
