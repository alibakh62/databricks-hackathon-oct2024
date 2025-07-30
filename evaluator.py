import os
import mlflow
import pandas as pd
from typing import List, Dict, Any, Optional
from mlflow.metrics.genai import make_genai_metric, EvaluationExample
from mlflow.metrics.genai import relevance, faithfulness, answer_relevance
import openai
from dotenv import load_dotenv

load_dotenv()

# Note: OpenAI API key is optional for initialization but required for evaluation


class CampaignEvaluator:
    """
    A class to evaluate campaign generation outputs using LLM-as-a-judge approach with MLflow.
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the evaluator with the specified model.
        
        Args:
            model_name: The OpenAI model to use for evaluation (default: gpt-4)
        """
        self.model_name = model_name
        
        # Initialize OpenAI client only if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "test_key":
            try:
                self.client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                self.client = None
        else:
            self.client = None
        
        # Set up MLflow experiment
        try:
            mlflow.set_experiment("/campaign-evaluation")
        except Exception as e:
            print(f"Warning: Could not set MLflow experiment: {e}")
        
        # Initialize custom metrics
        self._setup_custom_metrics()
    
    def _setup_custom_metrics(self):
        """Set up custom evaluation metrics for campaign generation."""
        
        # 1. Campaign Relevance Metric
        campaign_relevance_examples = [
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="This campaign focuses on the premium features and quality of our new coffee maker, targeting coffee enthusiasts who value craftsmanship and superior brewing technology.",
                score=5,
                justification="The output directly addresses the product and campaign goals, providing relevant insights for a coffee maker launch."
            ),
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="The weather forecast shows sunny days ahead with temperatures ranging from 65-75 degrees Fahrenheit.",
                score=1,
                justification="The output is completely irrelevant to the coffee maker campaign and product."
            ),
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="Coffee is a popular beverage consumed worldwide, with various brewing methods available.",
                score=3,
                justification="The output is somewhat related to coffee but doesn't specifically address the campaign or product features."
            )
        ]
        
        self.campaign_relevance = make_genai_metric(
            name="campaign_relevance",
            definition="Assesses how well the generated content aligns with the specific product and campaign objectives.",
            grading_prompt="Score from 1-5, where 1 is completely irrelevant and 5 is highly relevant to the campaign goals.",
            examples=campaign_relevance_examples
        )
        
        # 2. Marketing Effectiveness Metric
        marketing_effectiveness_examples = [
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="Our premium coffee maker features advanced brewing technology, customizable settings, and elegant design. Perfect for coffee enthusiasts who demand the best. Limited time offer: 20% off for early adopters!",
                score=5,
                justification="The output includes compelling product features, target audience identification, and a clear call-to-action with urgency."
            ),
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="We have a coffee maker. It makes coffee. You can buy it.",
                score=2,
                justification="The output lacks compelling marketing elements, emotional appeal, or clear value proposition."
            ),
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="Experience the perfect cup every time with our innovative coffee maker. Features include precision temperature control, programmable brewing, and a sleek stainless steel design that complements any kitchen.",
                score=4,
                justification="Good product description and features, but missing call-to-action and urgency elements."
            )
        ]
        
        self.marketing_effectiveness = make_genai_metric(
            name="marketing_effectiveness",
            definition="Evaluates the marketing appeal and persuasiveness of the generated content.",
            grading_prompt="Score from 1-5, where 1 is ineffective marketing and 5 is highly persuasive and compelling.",
            examples=marketing_effectiveness_examples
        )
        
        # 3. Research Quality Metric
        research_quality_examples = [
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="Market research shows that 65% of coffee drinkers prefer premium brewing equipment, with the specialty coffee market growing 8% annually. Our target demographic of 25-45 year olds values both quality and convenience.",
                score=5,
                justification="The output includes specific data points, market trends, and clear demographic targeting."
            ),
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="People like coffee. Coffee makers are popular. We should sell them.",
                score=1,
                justification="The output lacks any meaningful research, data, or market insights."
            ),
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="Coffee consumption is increasing globally, with many consumers seeking premium experiences.",
                score=3,
                justification="Basic market insight provided but lacks specific data or detailed analysis."
            )
        ]
        
        self.research_quality = make_genai_metric(
            name="research_quality",
            definition="Assesses the depth and quality of market research and insights provided.",
            grading_prompt="Score from 1-5, where 1 is poor research quality and 5 is excellent with detailed insights and data.",
            examples=research_quality_examples
        )
        
        # 4. Email Quality Metric
        email_quality_examples = [
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="Subject: Brew Perfection with Our Premium Coffee Maker - 20% Off!\n\nDear Coffee Enthusiast,\n\nDiscover the art of perfect brewing with our new premium coffee maker. Featuring advanced temperature control and customizable settings, it's designed for those who demand excellence in every cup.\n\nLimited Time Offer: Save 20% when you order this week!\n\n[Call-to-Action Button: Order Now]",
                score=5,
                justification="Excellent email structure with compelling subject line, personalization, clear value proposition, urgency, and strong call-to-action."
            ),
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="We have a coffee maker for sale. Buy it.",
                score=1,
                justification="Poor email structure, no personalization, weak value proposition, and no clear call-to-action."
            ),
            EvaluationExample(
                input="Product: Coffee maker, Campaign: Launch of new premium coffee maker",
                output="Subject: New Coffee Maker Available\n\nHello,\n\nWe have a new coffee maker. It makes coffee.\n\nThanks,\nThe Team",
                score=2,
                justification="Basic email structure but lacks compelling content, personalization, and effective marketing elements."
            )
        ]
        
        self.email_quality = make_genai_metric(
            name="email_quality",
            definition="Evaluates the quality and effectiveness of generated email marketing content.",
            grading_prompt="Score from 1-5, where 1 is poor email quality and 5 is excellent with compelling subject line, personalization, clear value proposition, and strong call-to-action.",
            examples=email_quality_examples
        )
    
    def evaluate_campaign_report(self, 
                                product_description: str, 
                                campaign_description: str, 
                                generated_report: str) -> Dict[str, Any]:
        """
        Evaluate a generated campaign report using LLM-as-a-judge metrics.
        
        Args:
            product_description: The original product description
            campaign_description: The original campaign description
            generated_report: The generated campaign report to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        
        # Prepare evaluation data
        eval_data = pd.DataFrame({
            "llm_inputs": [
                f"Product: {product_description}\nCampaign: {campaign_description}"
            ],
            "outputs": [generated_report]
        })
        
        # Define the model function (in this case, we're evaluating existing output)
        def dummy_model(inputs):
            # Return the outputs as-is since we're evaluating pre-generated content
            return inputs["outputs"].tolist()
        
        # Run evaluation with MLflow
        with mlflow.start_run() as run:
            results = mlflow.evaluate(
                dummy_model,
                eval_data,
                model_type="text",
                evaluators="default",
                extra_metrics=[
                    self.campaign_relevance,
                    self.marketing_effectiveness,
                    self.research_quality,
                    relevance(model=f"openai:/{self.model_name}"),
                    faithfulness(model=f"openai:/{self.model_name}"),
                    answer_relevance(model=f"openai:/{self.model_name}")
                ],
                evaluator_config={
                    "col_mapping": {
                        "inputs": "llm_inputs",
                        "outputs": "outputs"
                    }
                }
            )
        
        return {
            "run_id": run.info.run_id,
            "metrics": results.metrics,
            "artifacts": results.artifacts
        }
    
    def evaluate_email_content(self, 
                              product_description: str, 
                              campaign_description: str, 
                              generated_email: str) -> Dict[str, Any]:
        """
        Evaluate generated email content using LLM-as-a-judge metrics.
        
        Args:
            product_description: The original product description
            campaign_description: The original campaign description
            generated_email: The generated email content to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        
        # Prepare evaluation data
        eval_data = pd.DataFrame({
            "llm_inputs": [
                f"Product: {product_description}\nCampaign: {campaign_description}"
            ],
            "outputs": [generated_email]
        })
        
        # Define the model function
        def dummy_model(inputs):
            return inputs["outputs"].tolist()
        
        # Run evaluation with MLflow
        with mlflow.start_run() as run:
            results = mlflow.evaluate(
                dummy_model,
                eval_data,
                model_type="text",
                evaluators="default",
                extra_metrics=[
                    self.email_quality,
                    self.campaign_relevance,
                    self.marketing_effectiveness,
                    relevance(model=f"openai:/{self.model_name}"),
                    faithfulness(model=f"openai:/{self.model_name}"),
                    answer_relevance(model=f"openai:/{self.model_name}")
                ],
                evaluator_config={
                    "col_mapping": {
                        "inputs": "llm_inputs",
                        "outputs": "outputs"
                    }
                }
            )
        
        return {
            "run_id": run.info.run_id,
            "metrics": results.metrics,
            "artifacts": results.artifacts
        }
    
    def evaluate_complete_campaign(self, 
                                  product_description: str, 
                                  campaign_description: str, 
                                  generated_report: str, 
                                  generated_email: str) -> Dict[str, Any]:
        """
        Evaluate both the campaign report and email content together.
        
        Args:
            product_description: The original product description
            campaign_description: The original campaign description
            generated_report: The generated campaign report
            generated_email: The generated email content
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        
        # Evaluate report
        report_evaluation = self.evaluate_campaign_report(
            product_description, campaign_description, generated_report
        )
        
        # Evaluate email
        email_evaluation = self.evaluate_email_content(
            product_description, campaign_description, generated_email
        )
        
        # Combine results
        combined_metrics = {}
        combined_metrics.update(report_evaluation["metrics"])
        combined_metrics.update(email_evaluation["metrics"])
        
        # Calculate overall score (average of all metrics)
        numeric_metrics = [v for v in combined_metrics.values() if isinstance(v, (int, float))]
        overall_score = sum(numeric_metrics) / len(numeric_metrics) if numeric_metrics else 0
        
        return {
            "overall_score": overall_score,
            "report_evaluation": report_evaluation,
            "email_evaluation": email_evaluation,
            "combined_metrics": combined_metrics
        }
    
    def get_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of evaluation results.
        
        Args:
            evaluation_results: Results from evaluate_complete_campaign
            
        Returns:
            Formatted summary string
        """
        metrics = evaluation_results.get("combined_metrics", {})
        overall_score = evaluation_results.get("overall_score", 0)
        
        summary = f"""
# Campaign Evaluation Summary

## Overall Score: {overall_score:.2f}/5.0

## Detailed Metrics:

### Campaign Report Metrics:
- Campaign Relevance: {metrics.get('campaign_relevance', 'N/A')}/5.0
- Marketing Effectiveness: {metrics.get('marketing_effectiveness', 'N/A')}/5.0
- Research Quality: {metrics.get('research_quality', 'N/A')}/5.0
- Relevance: {metrics.get('relevance', 'N/A')}/5.0
- Faithfulness: {metrics.get('faithfulness', 'N/A')}/5.0
- Answer Relevance: {metrics.get('answer_relevance', 'N/A')}/5.0

### Email Content Metrics:
- Email Quality: {metrics.get('email_quality', 'N/A')}/5.0

## Recommendations:
"""
        
        # Add recommendations based on scores
        if overall_score < 3.0:
            summary += "- Overall quality needs significant improvement\n"
        elif overall_score < 4.0:
            summary += "- Good quality with room for improvement\n"
        else:
            summary += "- Excellent quality! Minor refinements may be beneficial\n"
        
        if metrics.get('campaign_relevance', 0) < 3.0:
            summary += "- Campaign content should be more aligned with product and campaign goals\n"
        
        if metrics.get('marketing_effectiveness', 0) < 3.0:
            summary += "- Marketing messaging needs to be more compelling and persuasive\n"
        
        if metrics.get('research_quality', 0) < 3.0:
            summary += "- Include more specific market data and insights\n"
        
        if metrics.get('email_quality', 0) < 3.0:
            summary += "- Email structure and content need improvement\n"
        
        return summary


def main():
    """Example usage of the CampaignEvaluator."""
    
    # Initialize evaluator
    evaluator = CampaignEvaluator()
    
    # Example data
    product_desc = "Premium coffee maker with advanced brewing technology"
    campaign_desc = "Launch campaign targeting coffee enthusiasts aged 25-45"
    
    # Example outputs (these would normally come from your campaign generation system)
    example_report = """
    # Market Analysis: Premium Coffee Maker Launch
    
    ## Market Overview
    The specialty coffee market is experiencing significant growth, with premium coffee makers seeing 12% annual growth. Our target demographic of 25-45 year olds represents 65% of premium coffee equipment purchases.
    
    ## Competitive Analysis
    Key competitors include Breville, DeLonghi, and Miele. Our unique selling proposition is advanced temperature control and customizable brewing profiles.
    
    ## Target Audience
    Primary: Coffee enthusiasts aged 25-45 with household income $60K+
    Secondary: Gift buyers for coffee lovers
    """
    
    example_email = """
    Subject: Brew Perfection Awaits - 20% Off Premium Coffee Maker!
    
    Dear Coffee Enthusiast,
    
    Discover the art of perfect brewing with our new premium coffee maker. Featuring advanced temperature control, customizable brewing profiles, and elegant stainless steel design.
    
    Limited Time Offer: Save 20% when you order this week!
    
    [Order Now] [Learn More]
    
    Best regards,
    The Coffee Team
    """
    
    # Run evaluation
    results = evaluator.evaluate_complete_campaign(
        product_desc, campaign_desc, example_report, example_email
    )
    
    # Print summary
    summary = evaluator.get_evaluation_summary(results)
    print(summary)
    
    return results


if __name__ == "__main__":
    main() 