# LLM-as-a-Judge Evaluation Module

This module implements an LLM-as-a-judge evaluation system using MLflow to assess the quality of campaign generation outputs from the MailGen application.

## Overview

The LLM-as-a-judge approach uses large language models to evaluate the quality of AI-generated content, providing more nuanced and context-aware assessments than traditional metrics. This implementation follows the best practices outlined in the [MLflow LLM-as-a-Judge blog post](https://www.mlflow.org/blog/llm-as-judge).

## Features

### Custom Evaluation Metrics

The system implements four custom metrics specifically designed for campaign evaluation:

1. **Campaign Relevance** - Assesses how well the generated content aligns with the specific product and campaign objectives
2. **Marketing Effectiveness** - Evaluates the marketing appeal and persuasiveness of the generated content
3. **Research Quality** - Assesses the depth and quality of market research and insights provided
4. **Email Quality** - Evaluates the quality and effectiveness of generated email marketing content

### Built-in MLflow Metrics

The system also leverages MLflow's built-in metrics:
- **Helpfulness** - Measures how helpful the content is to the user
- **Relevance** - Assesses the relevance of the content to the input
- **Coherence** - Evaluates the logical flow and consistency of the content

## Installation

1. Ensure you have the required dependencies:
```bash
pip install mlflow>=2.14.1 openai pandas python-dotenv
```

2. Set up your OpenAI API key in the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Usage

```python
from evaluator import CampaignEvaluator

# Initialize the evaluator
evaluator = CampaignEvaluator(model_name="gpt-4")

# Evaluate a complete campaign
results = evaluator.evaluate_complete_campaign(
    product_description="Premium coffee maker with advanced brewing technology",
    campaign_description="Launch campaign targeting coffee enthusiasts aged 25-45",
    generated_report="Your generated campaign report here...",
    generated_email="Your generated email content here..."
)

# Get a human-readable summary
summary = evaluator.get_evaluation_summary(results)
print(summary)
```

### Individual Component Evaluation

```python
# Evaluate just the campaign report
report_evaluation = evaluator.evaluate_campaign_report(
    product_description, campaign_description, generated_report
)

# Evaluate just the email content
email_evaluation = evaluator.evaluate_email_content(
    product_description, campaign_description, generated_email
)
```

### Integration with Gradio App

The evaluator is integrated into the main Gradio application. When you generate a campaign report and email, the system automatically:

1. Evaluates the generated content using LLM-as-a-judge metrics
2. Provides an overall quality score
3. Offers specific recommendations for improvement
4. Displays detailed metrics for each evaluation criterion

## Evaluation Process

### 1. Metric Definition

Each custom metric is defined with:
- **Definition**: Clear description of what the metric measures
- **Grading Prompt**: Instructions for the LLM judge on how to score
- **Examples**: Sample inputs, outputs, scores, and justifications

### 2. Evaluation Execution

The evaluation process:
1. Prepares evaluation data with inputs and outputs
2. Uses MLflow's `evaluate()` function with custom metrics
3. Leverages OpenAI's GPT models as judges
4. Returns structured results with scores and metadata

### 3. Result Analysis

Results include:
- Individual metric scores (1-5 scale)
- Overall composite score
- Detailed recommendations
- MLflow experiment tracking

## Customization

### Adding New Metrics

To add a new evaluation metric:

```python
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# Define examples
examples = [
    EvaluationExample(
        input="Your input example",
        output="Your output example",
        score=5,
        justification="Why this deserves a 5"
    ),
    # Add more examples...
]

# Create the metric
new_metric = make_genai_metric(
    name="your_metric_name",
    definition="What this metric measures",
    grading_prompt="How to score from 1-5",
    examples=examples
)
```

### Modifying Existing Metrics

You can modify the evaluation criteria by updating the examples and grading prompts in the `_setup_custom_metrics()` method of the `CampaignEvaluator` class.

## Testing

Run the test script to verify the evaluator works correctly:

```bash
python test_evaluator.py
```

This will:
- Test the complete evaluation pipeline
- Verify individual metrics work correctly
- Display sample results
- Check for any configuration issues

## MLflow Integration

The evaluator automatically:
- Creates MLflow experiments for tracking
- Logs evaluation runs with metrics and artifacts
- Provides experiment IDs for result tracking
- Enables comparison between different model outputs

## Best Practices

### 1. Example Quality
- Provide diverse examples covering different quality levels
- Include clear justifications for each score
- Use realistic inputs and outputs

### 2. Prompt Engineering
- Write clear, specific grading instructions
- Define the scoring scale explicitly
- Include context about the evaluation domain

### 3. Model Selection
- Use GPT-4 for highest quality evaluations
- Consider cost vs. quality trade-offs
- Test with different models for your use case

### 4. Result Interpretation
- Consider the overall score in context
- Pay attention to individual metric scores
- Use recommendations for iterative improvement

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Set**
   - Ensure `OPENAI_API_KEY` is in your `.env` file
   - Verify the key is valid and has sufficient credits

2. **MLflow Import Errors**
   - Install MLflow with GenAI support: `pip install mlflow>=2.14.1`
   - Ensure all dependencies are installed

3. **Evaluation Timeouts**
   - Consider using faster models (e.g., GPT-3.5-turbo)
   - Reduce the number of examples in metrics
   - Check your OpenAI rate limits

### Performance Optimization

- Use batch evaluation for multiple samples
- Cache evaluation results when possible
- Consider using smaller models for development/testing

## Contributing

To contribute to the evaluation system:

1. Add new metrics based on specific use cases
2. Improve example quality and diversity
3. Enhance the evaluation summary format
4. Add support for additional evaluation frameworks

## References

- [MLflow LLM-as-a-Judge Blog Post](https://www.mlflow.org/blog/llm-as-judge)
- [MLflow Evaluate Documentation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

## License

This module is part of the MailGen project and follows the same licensing terms. 