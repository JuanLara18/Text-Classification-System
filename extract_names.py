#!/usr/bin/env python3
"""
BMW Manufacturing Error Classification using ChatGPT
Analyzes sample data to determine optimal error categories based on actual production data.
"""

import os
import sys
import json
import pandas as pd
import openai
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

class BMWErrorClassifier:
    """
    Classifier for BMW manufacturing errors using OpenAI GPT models.
    Analyzes sample data to generate meaningful error categories.
    """
    
    def __init__(self, api_key_env: str = 'OPENAI_API_KEY'):
        """Initialize the classifier with OpenAI API key."""
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {api_key_env}")
        
        openai.api_key = self.api_key
        self.model = "gpt-4o-mini"  # Cost-effective model for classification
        self.results = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load BMW production data from Stata file."""
        try:
            print(f"Loading data from: {file_path}")
            df = pd.read_stata(file_path, convert_categoricals=False)
            print(f"Loaded {len(df)} records with {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def sample_data(self, df: pd.DataFrame, sample_size: int = 1500) -> pd.DataFrame:
        """Sample random records for analysis."""
        if len(df) <= sample_size:
            print(f"Using all {len(df)} records (less than requested sample size)")
            return df
        
        sample_df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {len(sample_df)} records from {len(df)} total records")
        return sample_df
    
    def prepare_sample_text(self, df: pd.DataFrame) -> str:
        """Prepare sample text from the 6 key variables for GPT analysis."""
        text_columns = [
            'CauseDescription_EN',
            'CauseLongText_EN', 
            'TechnicalObjectDescription_EN',
            'DamagePatternLongText_EN',
            'ShortDescription',  # German
            'Description'        # German
        ]
        
        # Check which columns exist
        available_columns = [col for col in text_columns if col in df.columns]
        if not available_columns:
            raise ValueError("No expected text columns found in the dataset")
        
        print(f"Using columns: {available_columns}")
        
        # Prepare sample texts
        sample_texts = []
        for idx, row in df.iterrows():
            record_texts = []
            for col in available_columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    record_texts.append(f"{col}: {str(row[col])[:200]}")  # Limit to 200 chars per field
            
            if record_texts:
                sample_texts.append(f"Record {len(sample_texts)+1}:\n" + "\n".join(record_texts))
        
        return "\n\n".join(sample_texts[:100])  # Limit to first 100 records for token management
    
    def create_analysis_prompt(self, sample_text: str, target_categories: int) -> str:
        """Create comprehensive prompt for GPT analysis."""
        
        prompt = f"""# BMW Manufacturing Error Classification Analysis

You are an expert in automotive manufacturing and quality control, specifically analyzing BMW production errors. 

## Task
Analyze the provided BMW manufacturing error data and create {target_categories} meaningful, distinct categories for classifying these errors. The categories should:

1. Be mutually exclusive and comprehensive
2. Reflect real manufacturing failure patterns
3. Be actionable for maintenance and quality teams
4. Follow industry best practices (FMEA, ISO standards)
5. Be suitable for both German and English text analysis

## Context
This data comes from BMW production facilities and includes:
- Cause descriptions (primary error descriptions)
- Technical object descriptions (equipment/component info)
- Damage patterns (how the failure manifested)
- Additional context in German and English

## Manufacturing Error Data Sample:
{sample_text}

## Requirements for {target_categories} Categories:

### Category Naming:
- Use clear, technical terminology
- Be specific enough to guide corrective action
- Follow automotive industry standards
- Avoid generic terms like "Other" or "Miscellaneous"

### Category Descriptions:
For each category, provide:
- **Name**: Clear, descriptive category name
- **Description**: What types of errors belong here
- **Typical Causes**: Common root causes
- **Examples**: 2-3 specific examples from the data

### Response Format:
```json
{{
  "analysis_summary": "Brief overview of error patterns found in the data",
  "categories": [
    {{
      "id": 1,
      "name": "Category Name",
      "description": "Detailed description of what errors belong in this category",
      "typical_causes": ["cause1", "cause2", "cause3"],
      "examples": ["example1", "example2"],
      "severity_level": "Critical/Major/Minor",
      "affected_systems": ["system1", "system2"]
    }}
  ],
  "classification_notes": "Additional insights about the classification approach"
}}
```

Generate exactly {target_categories} categories that best represent the error patterns in this BMW manufacturing data.
"""
        return prompt
    
    def analyze_with_gpt(self, prompt: str, category_count: int) -> Dict[str, Any]:
        """Send prompt to GPT and get classification analysis."""
        try:
            print(f"Analyzing data with GPT to generate {category_count} categories...")
            print(f"Prompt length: {len(prompt)} characters")
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert in automotive manufacturing, quality control, and FMEA analysis with deep knowledge of BMW production systems."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent, technical analysis
                max_tokens=3000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                # Find JSON in the response
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = result_text[start_idx:end_idx]
                    result_data = json.loads(json_str)
                    return result_data
                else:
                    raise ValueError("No JSON found in response")
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                print("Raw response:")
                print(result_text)
                
                # Return structured fallback
                return {
                    "analysis_summary": "GPT analysis completed but JSON parsing failed",
                    "categories": [],
                    "raw_response": result_text,
                    "error": str(e)
                }
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {"error": str(e)}
    
    def run_analysis(self, input_file: str, sample_size: int = 1500, 
                    category_counts: List[int] = [2, 7, 10, 20]) -> Dict[str, Any]:
        """
        Run complete analysis for multiple category counts.
        
        Args:
            input_file: Path to BMW data file
            sample_size: Number of records to sample
            category_counts: List of category counts to analyze
        """
        # Load and sample data
        df = self.load_data(input_file)
        sample_df = self.sample_data(df, sample_size)
        
        # Prepare sample text
        sample_text = self.prepare_sample_text(sample_df)
        print(f"Prepared sample text: {len(sample_text)} characters")
        
        # Run analysis for each category count
        results = {}
        total_cost = 0
        
        for count in category_counts:
            print(f"\n{'='*60}")
            print(f"ANALYZING FOR {count} CATEGORIES")
            print(f"{'='*60}")
            
            # Create prompt
            prompt = self.create_analysis_prompt(sample_text, count)
            
            # Analyze with GPT
            analysis_result = self.analyze_with_gpt(prompt, count)
            
            # Store result
            results[f"{count}_categories"] = {
                "category_count": count,
                "analysis": analysis_result,
                "timestamp": datetime.now().isoformat(),
                "sample_size": len(sample_df)
            }
            
            # Estimate cost (rough)
            estimated_tokens = len(prompt) // 4 + 1000  # Rough token estimate
            estimated_cost = estimated_tokens * 0.00015 / 1000  # GPT-4o-mini pricing
            total_cost += estimated_cost
            
            print(f"Analysis completed for {count} categories")
            print(f"Estimated cost: ${estimated_cost:.4f}")
            
            # Display categories if successful
            if "categories" in analysis_result and analysis_result["categories"]:
                print(f"\nGenerated Categories:")
                for i, cat in enumerate(analysis_result["categories"], 1):
                    if isinstance(cat, dict) and "name" in cat:
                        print(f"  {i}. {cat['name']}")
                        if "description" in cat:
                            print(f"     {cat['description'][:100]}...")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"bmw_error_classification_results_{timestamp}.json"
        
        final_results = {
            "metadata": {
                "input_file": input_file,
                "sample_size": len(sample_df),
                "total_records": len(df),
                "analysis_timestamp": datetime.now().isoformat(),
                "estimated_total_cost": total_cost,
                "model_used": self.model
            },
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"Total estimated cost: ${total_cost:.4f}")
        print(f"Sample size used: {len(sample_df)} of {len(df)} records")
        
        return final_results
    
    def display_results_summary(self, results: Dict[str, Any]):
        """Display a summary of all classification results."""
        print(f"\n{'='*80}")
        print(f"BMW ERROR CLASSIFICATION RESULTS SUMMARY")
        print(f"{'='*80}")
        
        metadata = results.get("metadata", {})
        print(f"Model: {metadata.get('model_used', 'Unknown')}")
        print(f"Sample Size: {metadata.get('sample_size', 'Unknown')}")
        print(f"Total Cost: ${metadata.get('estimated_total_cost', 0):.4f}")
        
        for key, result in results.get("results", {}).items():
            count = result.get("category_count", "Unknown")
            analysis = result.get("analysis", {})
            
            print(f"\n{'-'*40}")
            print(f"{count} CATEGORIES")
            print(f"{'-'*40}")
            
            if "analysis_summary" in analysis:
                print(f"Summary: {analysis['analysis_summary']}")
            
            categories = analysis.get("categories", [])
            for i, cat in enumerate(categories, 1):
                if isinstance(cat, dict):
                    name = cat.get("name", f"Category {i}")
                    desc = cat.get("description", "No description")
                    print(f"{i:2d}. {name}")
                    print(f"    {desc[:80]}...")


def main():
    """Main execution function."""
    # Configuration
    INPUT_FILE = "input/Maintenance_Classified_ErrorType.dta"  # Update path as needed
    SAMPLE_SIZE = 1500
    CATEGORY_COUNTS = [2, 7, 10, 20]
    
    print("BMW Manufacturing Error Classification Analysis")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        print("Please update the INPUT_FILE path in the script")
        sys.exit(1)
    
    # Check for OpenAI API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key as an environment variable")
        sys.exit(1)
    
    try:
        # Initialize classifier
        classifier = BMWErrorClassifier()
        
        # Run analysis
        results = classifier.run_analysis(
            input_file=INPUT_FILE,
            sample_size=SAMPLE_SIZE,
            category_counts=CATEGORY_COUNTS
        )
        
        # Display summary
        classifier.display_results_summary(results)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Check the generated JSON file for detailed results.")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()