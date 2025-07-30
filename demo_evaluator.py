#!/usr/bin/env python3
"""
Demonstration script for the LLM-as-a-judge evaluator.
This script shows how the evaluator works with sample data.
"""

import os
from evaluator import CampaignEvaluator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demo_evaluator():
    """Demonstrate the evaluator functionality."""
    
    print("🚀 LLM-as-a-Judge Evaluator Demonstration")
    print("="*60)
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "test_key":
        print("⚠️  No valid OpenAI API key found.")
        print("   The evaluator will be initialized but evaluation calls will fail.")
        print("   To run full evaluation, set OPENAI_API_KEY in your .env file.")
        print()
    
    try:
        # Initialize evaluator
        print("🔧 Initializing CampaignEvaluator...")
        evaluator = CampaignEvaluator(model_name="gpt-4")
        print("✅ Evaluator initialized successfully")
        
        # Sample data for demonstration
        product_desc = "Premium coffee maker with advanced brewing technology"
        campaign_desc = "Launch campaign targeting coffee enthusiasts aged 25-45"
        
        # Sample campaign report
        sample_report = """
        # Market Analysis: Premium Coffee Maker Launch
        
        ## Market Overview
        The specialty coffee market is experiencing significant growth, with premium coffee makers seeing 12% annual growth. Our target demographic of 25-45 year olds represents 65% of premium coffee equipment purchases.
        
        ## Competitive Analysis
        Key competitors include Breville, DeLonghi, and Miele. Our unique selling proposition is advanced temperature control and customizable brewing profiles.
        
        ## Target Audience
        Primary: Coffee enthusiasts aged 25-45 with household income $60K+
        Secondary: Gift buyers for coffee lovers
        """
        
        # Sample email content
        sample_email = """
        Subject: Brew Perfection Awaits - 20% Off Premium Coffee Maker!
        
        Dear Coffee Enthusiast,
        
        Discover the art of perfect brewing with our new premium coffee maker. Featuring advanced temperature control, customizable brewing profiles, and elegant stainless steel design.
        
        Limited Time Offer: Save 20% when you order this week!
        
        [Order Now] [Learn More]
        
        Best regards,
        The Coffee Team
        """
        
        print("\n📊 Sample Data:")
        print(f"Product: {product_desc}")
        print(f"Campaign: {campaign_desc}")
        print(f"Report Length: {len(sample_report)} characters")
        print(f"Email Length: {len(sample_email)} characters")
        
        print("\n🔍 Evaluation Metrics Available:")
        print("- Campaign Relevance")
        print("- Marketing Effectiveness") 
        print("- Research Quality")
        print("- Email Quality")
        print("- Relevance (MLflow built-in)")
        print("- Faithfulness (MLflow built-in)")
        print("- Answer Relevance (MLflow built-in)")
        
        if api_key and api_key != "test_key":
            print("\n📈 Running Full Evaluation...")
            print("(This will make API calls to OpenAI)")
            
            try:
                # Run evaluation
                results = evaluator.evaluate_complete_campaign(
                    product_desc, campaign_desc, sample_report, sample_email
                )
                
                # Display results
                summary = evaluator.get_evaluation_summary(results)
                print("\n" + "="*60)
                print("EVALUATION RESULTS")
                print("="*60)
                print(summary)
                
            except Exception as e:
                print(f"❌ Evaluation failed: {str(e)}")
                print("   This might be due to API limits or network issues.")
        else:
            print("\n📋 Evaluation Structure Demonstration:")
            print("The evaluator would:")
            print("1. Prepare evaluation data with inputs and outputs")
            print("2. Use MLflow's evaluate() function with custom metrics")
            print("3. Leverage OpenAI's GPT models as judges")
            print("4. Return structured results with scores and metadata")
            print("5. Generate human-readable summaries with recommendations")
            
            print("\n📝 Sample Evaluation Summary Format:")
            print("""
# Campaign Evaluation Summary

## Overall Score: 4.2/5.0

## Detailed Metrics:

### Campaign Report Metrics:
- Campaign Relevance: 4.5/5.0
- Marketing Effectiveness: 4.0/5.0
- Research Quality: 4.3/5.0
- Relevance: 4.2/5.0
- Faithfulness: 4.1/5.0
- Answer Relevance: 4.4/5.0

### Email Content Metrics:
- Email Quality: 4.0/5.0

## Recommendations:
- Good quality with room for improvement
- Include more specific market data and insights
            """)
        
        print("\n✅ Demonstration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_integration_info():
    """Show information about Gradio integration."""
    print("\n" + "="*60)
    print("GRADIO INTEGRATION")
    print("="*60)
    
    print("The evaluator is integrated into the main Gradio app:")
    print()
    print("1. When you generate a campaign report and email:")
    print("   - The system automatically evaluates the content")
    print("   - Provides an overall quality score")
    print("   - Offers specific recommendations for improvement")
    print("   - Displays detailed metrics for each criterion")
    print()
    print("2. New UI components added:")
    print("   - 'Show Evaluation' button appears after generation")
    print("   - Evaluation results displayed in markdown format")
    print("   - Clear button resets all outputs including evaluation")
    print()
    print("3. To use the integrated evaluation:")
    print("   - Run: python app.py")
    print("   - Go to 'Generate Report' tab")
    print("   - Enter product description, campaign description, and industry")
    print("   - Click 'Generate Report'")
    print("   - Click 'Show Evaluation' to see quality assessment")

def main():
    """Run the demonstration."""
    success = demo_evaluator()
    show_integration_info()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Set your OpenAI API key in .env file for full functionality")
    print("2. Run 'python app.py' to use the integrated Gradio interface")
    print("3. Run 'python test_evaluator.py' for full testing (requires API key)")
    print("4. Check LLM_AS_JUDGE_README.md for detailed documentation")
    
    return success

if __name__ == "__main__":
    main() 