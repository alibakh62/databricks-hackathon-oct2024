#!/usr/bin/env python3
"""
Test script for the LLM-as-a-judge evaluator.
This script tests the CampaignEvaluator class with sample data.
"""

import os
from evaluator import CampaignEvaluator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_evaluator():
    """Test the CampaignEvaluator with sample data."""
    
    # Check if OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in the .env file")
        return False
    
    print("✅ OpenAI API key found")
    
    try:
        # Initialize evaluator
        print("🔧 Initializing CampaignEvaluator...")
        evaluator = CampaignEvaluator(model_name="gpt-4")
        print("✅ Evaluator initialized successfully")
        
        # Test data
        product_desc = "Premium coffee maker with advanced brewing technology and customizable settings"
        campaign_desc = "Launch campaign targeting coffee enthusiasts aged 25-45 with focus on quality and convenience"
        
        # Example campaign report
        example_report = """
        # Market Analysis: Premium Coffee Maker Launch
        
        ## Market Overview
        The specialty coffee market is experiencing significant growth, with premium coffee makers seeing 12% annual growth. Our target demographic of 25-45 year olds represents 65% of premium coffee equipment purchases.
        
        ## Competitive Analysis
        Key competitors include Breville, DeLonghi, and Miele. Our unique selling proposition is advanced temperature control and customizable brewing profiles that cater to individual preferences.
        
        ## Target Audience
        Primary: Coffee enthusiasts aged 25-45 with household income $60K+
        Secondary: Gift buyers for coffee lovers
        
        ## Market Trends
        - Growing demand for home brewing equipment
        - Increasing interest in specialty coffee
        - Preference for smart, connected appliances
        """
        
        # Example email content
        example_email = """
        Subject: Brew Perfection Awaits - 20% Off Premium Coffee Maker!
        
        Dear Coffee Enthusiast,
        
        Discover the art of perfect brewing with our new premium coffee maker. Featuring advanced temperature control, customizable brewing profiles, and elegant stainless steel design that complements any kitchen.
        
        Why choose our premium coffee maker?
        • Precision temperature control for optimal extraction
        • Customizable brewing profiles for different coffee types
        • Smart connectivity for remote brewing
        • Elegant design that enhances your kitchen aesthetic
        
        Limited Time Offer: Save 20% when you order this week!
        
        [Order Now] [Learn More] [View Demo]
        
        Best regards,
        The Coffee Team
        """
        
        print("\n📊 Running evaluation...")
        
        # Run evaluation
        results = evaluator.evaluate_complete_campaign(
            product_desc, campaign_desc, example_report, example_email
        )
        
        print("✅ Evaluation completed successfully")
        
        # Generate and display summary
        summary = evaluator.get_evaluation_summary(results)
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(summary)
        
        # Display detailed metrics
        print("\n" + "="*50)
        print("DETAILED METRICS")
        print("="*50)
        metrics = results.get("combined_metrics", {})
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score}/5.0")
        
        print(f"\nOverall Score: {results.get('overall_score', 0):.2f}/5.0")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_metrics():
    """Test individual evaluation metrics."""
    
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL METRICS")
    print("="*50)
    
    try:
        evaluator = CampaignEvaluator()
        
        # Test campaign relevance
        print("\n🔍 Testing Campaign Relevance...")
        relevance_result = evaluator.evaluate_campaign_report(
            "Coffee maker",
            "Launch campaign",
            "This is a comprehensive analysis of the coffee market and our product positioning."
        )
        print(f"Campaign Relevance Score: {relevance_result['metrics'].get('campaign_relevance', 'N/A')}")
        
        # Test email quality
        print("\n📧 Testing Email Quality...")
        email_result = evaluator.evaluate_email_content(
            "Coffee maker",
            "Launch campaign",
            "Subject: New Coffee Maker\n\nDear Customer,\n\nWe have a new coffee maker available. Order now!"
        )
        print(f"Email Quality Score: {email_result['metrics'].get('email_quality', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing individual metrics: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting LLM-as-a-Judge Evaluator Tests")
    print("="*50)
    
    # Run main test
    success1 = test_evaluator()
    
    # Run individual metrics test
    success2 = test_individual_metrics()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The evaluator is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.") 