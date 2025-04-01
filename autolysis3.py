import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# Load API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY is not set. Story generation will be skipped.")
    api_key = None
else:
    genai.configure(api_key=api_key)

# Function to load dataset
def load_data(file_path):
    """Loads a CSV file and returns a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, encoding="latin1", on_bad_lines="skip")  # Handle encoding issues
        print(f"Dataset '{file_path}' loaded successfully!")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Dataset Shape: {df.shape}\n")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

# Function for basic analysis
def basic_analysis(df):
    """Performs basic analysis on the dataset."""
    print("Basic Analysis:")
    print("===============")
    print(df.head(), "\n")
    print("Summary Statistics:\n", df.describe(include="all"), "\n")
    print("Missing Values Count:\n", df.isnull().sum(), "\n")

# Function for generating visualizations
def generate_visualizations(df, output_dir="happiness_media"):
    """Generates and saves visualizations for the dataset."""
    print("Generating Visualizations...")
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    if "Happiness Score" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Happiness Score"].dropna(), bins=30, kde=True)
        plt.title("Distribution of Happiness Scores")
        plt.xlabel("Happiness Score")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/happiness_distribution.png")
        print("Saved: happiness_distribution.png")

    if set(["Happiness Score", "GDP per Capita", "Social Support", "Healthy Life Expectancy", 
            "Freedom to Make Life Choices", "Generosity", "Perceptions of Corruption"]).issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[["Happiness Score", "GDP per Capita", "Social Support", "Healthy Life Expectancy", 
                        "Freedom to Make Life Choices", "Generosity", "Perceptions of Corruption"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        print("Saved: correlation_heatmap.png")

# Function to generate story using Gemini
def generate_story_gemini(df, output_dir="happiness_media"):
    """Uses Gemini AI to generate a story based on data analysis."""
    if not api_key:
        print("Skipping story generation due to missing API key.")
        return
    
    print("Generating Story...")
    os.makedirs(output_dir, exist_ok=True)
    
    avg_happiness = df["Happiness Score"].mean() if "Happiness Score" in df.columns else None
    avg_happiness_str = f"{avg_happiness:.2f}" if isinstance(avg_happiness, (int, float)) else "N/A"
    
    top_countries = df.nlargest(3, "Happiness Score")["Country"].tolist() if "Happiness Score" in df.columns and "Country" in df.columns else []
    bottom_countries = df.nsmallest(3, "Happiness Score")["Country"].tolist() if "Happiness Score" in df.columns and "Country" in df.columns else []
    
    missing_values = df.isnull().sum().sum()

    prompt = f"""
    You are an AI data analyst. Summarize the following world happiness dataset analysis as a compelling story:
    - The dataset contains {df.shape[0]} countries with {df.shape[1]} attributes.
    - The average happiness score is {avg_happiness_str}.
    - The top 3 happiest countries are: {', '.join(top_countries) if top_countries else 'Unknown'}.
    - The bottom 3 least happy countries are: {', '.join(bottom_countries) if bottom_countries else 'Unknown'}.
    - Missing values: {missing_values} across multiple fields.

    Provide insights on happiness trends, key influencing factors, and interesting patterns.
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        story = response.text if hasattr(response, "text") else "No response generated."
        
        readme_path = f"{output_dir}/README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# World Happiness Dataset Analysis\n\n")
            f.write(story)
        
        print(f"Story saved to {readme_path}")
    except Exception as e:
        print(f"Error generating story: {e}")

# Main Execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autolysis_happiness.py happiness.csv")
        sys.exit(1)

    dataset_path = sys.argv[1]
    df = load_data(dataset_path)

    basic_analysis(df)
    generate_visualizations(df, "happiness_media")
    generate_story_gemini(df, "happiness_media")