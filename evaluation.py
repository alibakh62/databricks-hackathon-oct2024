"""Utilities for collecting human feedback and running automated evaluations."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


@dataclass
class FeedbackEntry:
    """Container for a single feedback record."""

    timestamp: str
    product_description: str
    campaign_description: str
    industry: str
    conversation: List[Dict[str, str]]
    user_rating: Optional[int] = None
    user_comments: Optional[str] = None
    automated_evaluation: Optional[Dict[str, str]] = None


class FeedbackLogger:
    """Persist feedback entries to disk for later analysis."""

    def __init__(self, base_dir: str = "evaluations") -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_feedback(self, entry: FeedbackEntry) -> str:
        """Persist a feedback entry to disk and return the saved path."""

        timestamp = entry.timestamp.replace(":", "-")
        file_name = f"feedback_{timestamp}.json"
        file_path = os.path.join(self.base_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(entry), f, ensure_ascii=False, indent=2)
        return file_name

    def list_feedback_files(self) -> List[str]:
        """Return a sorted list of all stored feedback artifacts."""

        files = [
            file_name
            for file_name in os.listdir(self.base_dir)
            if file_name.endswith(".json")
        ]
        return sorted(files)

    def load_feedback(self, file_name: str) -> Dict[str, object]:
        """Load a feedback artifact into memory."""

        file_path = os.path.join(self.base_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


def build_feedback_entry(
    conversation: List[Dict[str, str]],
    product_description: str,
    campaign_description: str,
    industry: str,
    rating: Optional[int],
    comments: Optional[str],
    automated_evaluation: Optional[Dict[str, str]] = None,
) -> FeedbackEntry:
    """Create a feedback entry with a normalized timestamp."""

    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return FeedbackEntry(
        timestamp=timestamp,
        product_description=product_description,
        campaign_description=campaign_description,
        industry=industry,
        conversation=conversation,
        user_rating=rating,
        user_comments=comments,
        automated_evaluation=automated_evaluation,
    )


def run_quality_evaluation(
    conversation: List[Dict[str, str]],
    product_description: str,
    campaign_description: str,
    industry: str,
) -> Dict[str, str]:
    """Use an LLM to score the conversation quality and surface action items."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert marketing operations leader tasked with evaluating an
                AI assistant that designs email campaigns. Review the conversation between
                the marketer and the assistant. Provide constructive feedback, numerical
                quality scores (1-5 scale), and identify concrete next actions for the
                assistant to improve the campaign. Return a JSON object that matches the
                following schema:
                {"alignment_score": "1-5 score as string",
                 "clarity_score": "1-5 score as string",
                 "actionability_score": "1-5 score as string",
                 "summary": "one paragraph summary of the assistant performance",
                 "next_steps": ["Ordered list of next actions for the assistant"]}
                """,
            ),
            (
                "human",
                """Conversation history:\n{conversation}\n\nProduct description: {product}\nCampaign description: {campaign}\nIndustry: {industry}\n\nProvide the evaluation now.""",
            ),
        ]
    )

    formatted_conversation = json.dumps(conversation, ensure_ascii=False, indent=2)

    chain = prompt | llm | parser
    return chain.invoke(
        {
            "conversation": formatted_conversation,
            "product": product_description,
            "campaign": campaign_description,
            "industry": industry,
        }
    )
