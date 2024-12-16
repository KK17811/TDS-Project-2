import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def perform_analysis(df):
    """Perform comprehensive generic analysis."""
    analysis = {
        "basic_info": {
            "shape": df.shape,
            "columns": list(df.columns),
            "column_types": str(df.dtypes)
        },
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include="all").to_dict()
    }
    
    # Correlation for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        analysis["correlation_matrix"] = df[numeric_cols].corr().to_dict()
    
    return analysis

def create_visualizations(df, prefix):
    """Generate informative visualizations."""
    charts = []
    plt.figure(figsize=(5.12, 5.12))
    
    # Correlation Heatmap
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(5.12, 5.12))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        correlation_file = f"{prefix}_correlation_heatmap.png"
        plt.savefig(correlation_file, bbox_inches='tight')
        charts.append(correlation_file)
        plt.close()
    
    # Distribution of first numeric column
    if len(numeric_cols) > 0:
        plt.figure(figsize=(5.12, 5.12))
        sns.histplot(df[numeric_cols[0]], kde=True)
        plt.title(f"Distribution of {numeric_cols[0]}")
        distribution_file = f"{prefix}_distribution.png"
        plt.savefig(distribution_file, bbox_inches='tight')
        charts.append(distribution_file)
        plt.close()
    
    # First categorical column value counts
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        plt.figure(figsize=(5.12, 5.12))
        df[categorical_cols[0]].value_counts().head(10).plot(kind='bar')
        plt.title(f"Top 10 {categorical_cols[0]} Categories")
        plt.xticks(rotation=45, ha='right')
        categories_file = f"{prefix}_categories.png"
        plt.savefig(categories_file, bbox_inches='tight')
        charts.append(categories_file)
        plt.close()
    
    return charts

def generate_narrative(analysis, charts, filename):
    """Generate a narrative using AI Proxy and GPT-4o-Mini."""
    aiproxy_token = os.getenv("AIPROXY_TOKEN")
    if not aiproxy_token:
        print("Error: AIPROXY_TOKEN not set")
        sys.exit(1)

    prompt = f"""Analyze this dataset from {filename}:

Dataset Overview:
- Total Rows: {analysis['basic_info']['shape'][0]}
- Total Columns: {analysis['basic_info']['shape'][1]}
- Columns: {', '.join(analysis['basic_info']['columns'])}
- Column Types: {analysis['basic_info']['column_types']}

Missing Values Summary:
{analysis['missing_values']}

Key Insights From Analysis:
{analysis['summary_statistics']}

Write a compelling narrative that:
1. Describes the dataset briefly
2. Highlights interesting findings
3. Provides potential insights or recommendations
4. Explains implications of the data

Be creative, engaging, and use a storytelling approach. Make it sound like an investigative data journalism piece."""

    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {aiproxy_token}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7
            }
        )
        
        # Check for successful response
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        # Parse response
        result = response.json()
        narrative = result['choices'][0]['message']['content']
        
        # Print cost information
        print(f"Request Cost: ${response.headers.get('cost', 'N/A')}")
        print(f"Monthly Cost: ${response.headers.get('monthlyCost', 'N/A')}")
        print(f"Monthly Requests: {response.headers.get('monthlyRequests', 'N/A')}")
        
    except Exception as e:
        narrative = f"Error generating narrative: {e}"
    
    return narrative

def main(input_file,df):
    """Main processing workflow."""
    # Ensure output directory exists
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(dataset_name, exist_ok=True)
    os.chdir(dataset_name)
    
    # Read and analyze CSV
    analysis = perform_analysis(df)
    
    # Create visualizations
    charts = create_visualizations(df, dataset_name)
    
    # Generate narrative
    narrative = generate_narrative(analysis, charts, input_file)
    
    # Save README
    with open("README.md", "w") as f:
        f.write("# Dataset Analysis\n\n")
        f.write(narrative + "\n\n")
        for chart in charts:
            f.write(f"![{chart}]({chart})\n")
    
    print(f"Analysis complete for {input_file}")

if __name__ == "__main__":
    # Expect CSV file as command-line argument
    if len(sys.argv) != 2:
        print("Usage: python ans.py <input.csv>")
        sys.exit(1)
    
    df = pd.read_csv(sys.argv[1], encoding="latin1")
    main(sys.argv[1],df)