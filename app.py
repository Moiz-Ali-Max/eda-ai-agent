import os, io, asyncio, tempfile, traceback, time
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import chainlit as cl
from PIL import Image
import google.generativeai as genai 
import matplotlib
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from datetime import datetime
matplotlib.use('Agg')

# Enhanced styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Professional color palette
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
# Plotly theme
PROFESSIONAL_THEME = 'plotly_white'

GEMINI_MODEL = "gemini-2.0-flash"

GEMINI_AVAILABLE = False

try:
    if api_key := os.environ.get("GEMINI_API_KEY"):
        genai.configure(api_key = api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        GEMINI_AVAILABLE = True
except Exception as e:
    print(f" Gemini init failed: {e}")

def save_fig(fig):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(f.name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return f.name

def save_plotly_fig(fig, filename_prefix="plot"):
    """Save plotly figure as HTML and convert to image"""
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(f.name)
    return f.name

def comprehensive_data_overview(df):
    """Generate comprehensive data overview with statistics"""
    overview = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        overview['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        overview['categorical_summary'] = {}
        for col in categorical_cols:
            overview['categorical_summary'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head().to_dict()
            }
    
    return overview

def format_data_overview(overview, df):
    """Format data overview into markdown"""
    md = f"""# 📊 **DATA ANALYSIS REPORT**
    
## 📋 **Dataset Overview**
- **Dimensions:** {overview['shape'][0]:,} rows × {overview['shape'][1]} columns
- **Memory Usage:** {overview['memory_usage'] / 1024 / 1024:.2f} MB
- **Duplicate Rows:** {overview['duplicate_rows']:,}

## 📈 **Data Quality Assessment**
"""
    
    # Missing values analysis
    missing_data = [(k, v) for k, v in overview['missing_values'].items() if v > 0]
    if missing_data:
        md += "### ⚠️ **Missing Values Detected:**\n"
        for col, count in missing_data:
            percentage = (count / overview['shape'][0]) * 100
            md += f"- **{col}:** {count:,} values ({percentage:.2f}%)\n"
    else:
        md += "### ✅ **Data Completeness:** No missing values detected\n"
    
    # Data types
    md += f"\n### 📊 **Column Types:**\n"
    type_counts = {}
    for col, dtype in overview['dtypes'].items():
        dtype_str = str(dtype)
        type_counts[dtype_str] = type_counts.get(dtype_str, 0) + 1
    
    for dtype, count in type_counts.items():
        md += f"- **{dtype}:** {count} columns\n"
    
    # Sample data
    md += f"\n### 🔍 **Sample Data:**\n"
    md += df.head(3).to_markdown(index=False)
    
    return md


async def enhanced_ai_analysis(prompt_type, context_data, chart_type=None):
    """Enhanced AI analysis with specific prompts for different types of analysis"""
    if not GEMINI_AVAILABLE: 
        return f"🤖 Gemini AI not available. Please set GEMINI_API_KEY environment variable."

    prompts = {
        "executive_summary": f"""
As a senior data analyst, provide a comprehensive executive summary of this dataset:

{context_data}

Please structure your response with:
1. 📊 **Key Dataset Metrics**
2. 🔍 **Data Quality Assessment** 
3. 💡 **Primary Insights**
4. 🚨 **Critical Findings**
5. 📈 **Business Recommendations**

Use professional language with emojis and make it actionable.
""",
        
        "correlation_insights": f"""
Analyze this correlation matrix data and provide insights:

{context_data}

Focus on:
- Strong positive/negative correlations (>0.7 or <-0.7)
- Business implications of these relationships
- Potential causality vs correlation warnings
- Actionable recommendations

Format with emojis and bullet points.
""",
        
        "distribution_insights": f"""
Analyze the distribution patterns in this data:

{context_data}

Provide insights on:
- Distribution shapes (normal, skewed, bimodal)
- Presence of outliers and their implications
- Central tendencies and variability
- Data transformation recommendations if needed

Use professional terminology with practical recommendations.
""",
        
        "categorical_insights": f"""
Analyze these categorical patterns:

{context_data}

Focus on:
- Dominant categories and their business significance
- Distribution balance and potential bias
- Rare categories that might need special attention
- Strategic recommendations for category management
""",
        
        "outlier_insights": f"""
Analyze the outlier patterns shown:

{context_data}

Provide analysis on:
- Severity and frequency of outliers
- Potential causes (data errors vs genuine extreme values)
- Impact on analysis and modeling
- Recommended treatment strategies
""",
        
        "relationship_insights": f"""
Analyze this relationship analysis:

{context_data}

Focus on:
- Strength and direction of relationships
- Linear vs non-linear patterns
- Business implications of these relationships
- Predictive modeling opportunities
""",
        
        "final_report": f"""
As a senior data scientist, create a comprehensive final analysis report:

{context_data}

Structure your response as:

# 📋 **EXECUTIVE SUMMARY**
[Key findings and recommendations]

# 🔬 **DETAILED ANALYSIS**
[Technical insights and patterns]

# 💼 **BUSINESS IMPLICATIONS**
[How these insights impact business decisions]

# 🚀 **NEXT STEPS**
[Recommended actions and further analysis]

# ⚠️ **LIMITATIONS & CONSIDERATIONS**
[Data limitations and analytical caveats]

Make it professional, actionable, and insightful.
"""
    }

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        res = await model.generate_content_async(
            prompts.get(prompt_type, prompts["executive_summary"]), 
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=800, temperature=0.3
            )
        )
        return res.text if res.parts else "🤖 Gemini response was blocked."
    except Exception as e:
        return f"🚫 Gemini error: {e}"

async def ai_vision_analysis(img_paths):
    if not GEMINI_AVAILABLE: return [("AI Vision", "Gemini not available.")]

    model = genai.GenerativeModel(GEMINI_MODEL) 
    results = []

    for title, path in img_paths:
        try:
            img = Image.open(path)
            res = await model.generate_content_async([f"Explain this '{title}'", img],
                                                     generation_config=genai.types.GenerationConfig(
                                                         max_output_tokens=300, temperature=0.3))
            results.append((title, res.text if res.parts else " Gemini response blocked."))
        except Exception as e:
            results.append((title, f"Error: {e}"))
    return results


def create_interactive_correlation_heatmap(df):
    """Create an interactive correlation heatmap using Plotly"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title={
            'text': '🔥 Interactive Correlation Matrix',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial Black'}
        },
        template=PROFESSIONAL_THEME,
        width=800,
        height=600,
        font={'size': 12}
    )
    
    return fig

def create_distribution_analysis(df):
    """Create distribution plots for numeric variables"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4 for readability
    if len(numeric_cols) == 0:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"📈 Distribution of {col}" for col in numeric_cols[:4]],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = COLOR_PALETTE[:len(numeric_cols)]
    
    for i, col in enumerate(numeric_cols):
        if i < 4:  # Safety check
            row, col_pos = positions[i]
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    name=col,
                    marker_color=colors[i],
                    opacity=0.7,
                    nbinsx=30
                ),
                row=row, col=col_pos
            )
            
            # Add mean line
            mean_val = df[col].mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}",
                row=row, col=col_pos
            )
    
    fig.update_layout(
        title={
            'text': '📊 Advanced Distribution Analysis',
            'x': 0.5,
            'font': {'size': 20, 'family': 'Arial Black'}
        },
        template=PROFESSIONAL_THEME,
        height=700,
        showlegend=False
    )
    
    return fig

def create_categorical_analysis(df):
    """Create categorical variable analysis"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if df[col].nunique() <= 20 and df[col].nunique() > 1]
    
    if len(categorical_cols) == 0:
        return None
    
    # Take first categorical column for detailed analysis
    col = categorical_cols[0]
    value_counts = df[col].value_counts().head(15)
    
    # Create animated bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=value_counts.values,
        y=value_counts.index,
        orientation='h',
        marker=dict(
            color=value_counts.values,
            colorscale='viridis',
            line=dict(color='rgba(50,50,50,0.8)', width=1)
        ),
        text=[f'{v:,} ({v/len(df)*100:.1f}%)' for v in value_counts.values],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Count: %{x:,}<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=[v/len(df)*100 for v in value_counts.values]
    ))
    
    fig.update_layout(
        title={
            'text': f'🏷️ Top Categories in {col}',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial Black'}
        },
        template=PROFESSIONAL_THEME,
        height=600,
        xaxis_title='Count',
        yaxis_title=col,
        font={'size': 12}
    )
    
    return fig

def create_outlier_analysis(df):
    """Create outlier detection visualization"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
    if len(numeric_cols) == 0:
        return None
    
    fig = go.Figure()
    
    for i, col in enumerate(numeric_cols):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            boxpoints='outliers',
            marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            line_color=COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ))
    
    fig.update_layout(
        title={
            'text': '🎯 Outlier Detection Analysis',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial Black'}
        },
        template=PROFESSIONAL_THEME,
        height=600,
        yaxis_title='Values',
        font={'size': 12}
    )
    
    return fig

def create_advanced_scatter_analysis(df):
    """Create advanced scatter plot with trend lines"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    # Get two most correlated variables
    corr_matrix = df[numeric_cols].corr()
    
    # Find the pair with highest absolute correlation (excluding self-correlation)
    corr_matrix_abs = corr_matrix.abs()
    np.fill_diagonal(corr_matrix_abs.values, 0)
    
    if corr_matrix_abs.max().max() > 0:
        max_corr_idx = corr_matrix_abs.stack().idxmax()
        x_col, y_col = max_corr_idx
    else:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
    
    fig = px.scatter(
        df, x=x_col, y=y_col,
        trendline="ols",
        title=f"💫 Advanced Relationship Analysis: {x_col} vs {y_col}",
        template=PROFESSIONAL_THEME,
        color_discrete_sequence=COLOR_PALETTE
    )
    
    fig.update_layout(
        title={
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial Black'}
        },
        height=600,
        font={'size': 12}
    )
    
    return fig

def generate_advanced_visualizations(df):
    """Generate all advanced visualizations"""
    visualizations = []
    saved_files = []
    
    try:
        # 1. Interactive Correlation Heatmap
        corr_fig = create_interactive_correlation_heatmap(df)
        if corr_fig:
            path = save_plotly_fig(corr_fig, "correlation_heatmap")
            visualizations.append(("🔥 Interactive Correlation Matrix", path))
            saved_files.append(path)
        
        # 2. Distribution Analysis
        dist_fig = create_distribution_analysis(df)
        if dist_fig:
            path = save_plotly_fig(dist_fig, "distribution_analysis")
            visualizations.append(("📊 Advanced Distribution Analysis", path))
            saved_files.append(path)
        
        # 3. Categorical Analysis
        cat_fig = create_categorical_analysis(df)
        if cat_fig:
            path = save_plotly_fig(cat_fig, "categorical_analysis")
            visualizations.append(("🏷️ Categorical Data Analysis", path))
            saved_files.append(path)
        
        # 4. Outlier Analysis
        outlier_fig = create_outlier_analysis(df)
        if outlier_fig:
            path = save_plotly_fig(outlier_fig, "outlier_analysis")
            visualizations.append(("🎯 Outlier Detection", path))
            saved_files.append(path)
        
        # 5. Advanced Scatter Analysis
        scatter_fig = create_advanced_scatter_analysis(df)
        if scatter_fig:
            path = save_plotly_fig(scatter_fig, "scatter_analysis")
            visualizations.append(("💫 Relationship Analysis", path))
            saved_files.append(path)
    
    except Exception as e:
        print(f"Advanced Visualization Error: {e}")
    
    return visualizations, saved_files


async def cleanup(files):
    for f in files:
        try: os.remove(f)
        except: pass


@cl.on_chat_start
async def start():
    # Professional welcome message
    welcome_msg = """
# 🎆 **PROFESSIONAL EDA AI AGENT**

🔬 **Advanced Data Analytics Platform**  
🎯 **Powered by Gemini AI & Interactive Visualizations**


**Please upload your CSV file to begin the analysis...**
    """
    
    await cl.Message(content=welcome_msg).send()
    
    files = await cl.AskFileMessage(
        content="📄 **Upload your CSV file for professional analysis**", 
        accept=["text/csv", "application/vnd.ms-excel"]
    ).send()

    if not files:
        return await cl.Message(content="❌ No file received. Please try again.").send()
    
    # Enhanced progress tracking
    processing_msg = cl.Message(content="🔄 **Phase 1:** Reading and validating data...")
    await processing_msg.send()

    try:
        # File reading with enhanced error handling
        file_path = files[0].path
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                await asyncio.sleep(0.5)  # Small delay for better UX
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if df is None:
            raise ValueError("Could not read file with any common encoding")

        if df.empty:
            processing_msg.content = "⚠️ Dataset is empty. Please upload a valid CSV file."
            await processing_msg.update()
            return
        
        cl.user_session.set("df", df)
        
        # Phase 2: Data Overview
        processing_msg.content = "🔄 **Phase 2:** Generating comprehensive data overview..."
        await processing_msg.update()
        await asyncio.sleep(0.5)
        
        overview = comprehensive_data_overview(df)
        formatted_overview = format_data_overview(overview, df)
        await cl.Message(content=formatted_overview).send()

        # Phase 3: Executive Summary (AI Analysis)
        if GEMINI_AVAILABLE:
            processing_msg.content = "🔄 **Phase 3:** Generating AI-powered executive summary..."
            await processing_msg.update()
            await asyncio.sleep(0.5)
            
            executive_summary = await enhanced_ai_analysis("executive_summary", formatted_overview)
            await cl.Message(content=f"# 📋 **EXECUTIVE SUMMARY**\n\n{executive_summary}").send()

        # Phase 4: Advanced Visualizations
        processing_msg.content = "🔄 **Phase 4:** Creating advanced interactive visualizations..."
        await processing_msg.update()
        await asyncio.sleep(0.5)

        visuals, saved_files = generate_advanced_visualizations(df)
        
        # Display each visualization with detailed analysis
        visualization_types = {
            "🔥 Interactive Correlation Matrix": "correlation_insights",
            "📊 Advanced Distribution Analysis": "distribution_insights",
            "🏷️ Categorical Data Analysis": "categorical_insights",
            "🎯 Outlier Detection": "outlier_insights",
            "💫 Relationship Analysis": "relationship_insights"
        }
        
        for title, path in visuals:
            # Display visualization
            if path.endswith('.html'):
                # For HTML files (Plotly), create a file element
                with open(path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                await cl.Message(
                    content=f"## {title}",
                    elements=[cl.File(name=f"{title}.html", path=path, display="inline")]
                ).send()
            else:
                # For image files
                await cl.Message(
                    content=f"## {title}",
                    elements=[cl.Image(name=title, path=path)]
                ).send()
            
            # AI Analysis for each visualization
            if GEMINI_AVAILABLE and title in visualization_types:
                processing_msg.content = f"🤖 Analyzing {title}..."
                await processing_msg.update()
                
                analysis_type = visualization_types[title]
                insight = await enhanced_ai_analysis(analysis_type, formatted_overview)
                await cl.Message(content=f"### 💡 **AI Insights for {title}**\n\n{insight}").send()
                
                await asyncio.sleep(0.3)  # Brief pause between analyses

        # Phase 5: Final Comprehensive Report
        if GEMINI_AVAILABLE:
            processing_msg.content = "🔄 **Phase 5:** Generating comprehensive final report..."
            await processing_msg.update()
            await asyncio.sleep(0.5)
            
            final_report = await enhanced_ai_analysis("final_report", formatted_overview)
            await cl.Message(content=f"# 📋 **COMPREHENSIVE ANALYSIS REPORT**\n\n{final_report}").send()

        # Completion message
        completion_msg = """
# 🎉 **ANALYSIS COMPLETE!**

✅ **Professional EDA Analysis Finished**

## 📈 **What You Received:**
- Comprehensive data overview and quality assessment
- Interactive correlation analysis
- Advanced statistical distribution analysis
- Professional outlier detection
- Categorical data profiling
- Relationship analysis with trend lines
- AI-powered insights and recommendations
- Executive summary and comprehensive report

### 🚀 **Next Steps:**
- Review the insights and recommendations
- Consider the business implications
- Plan further analysis based on findings
- Implement data-driven decisions

**Thank you for using our Professional EDA AI Agent!**
        """
        
        processing_msg.content = completion_msg
        await processing_msg.update()
        await cleanup(saved_files)

    except Exception as e:
        error_msg = f"""
# ❌ **ERROR OCCURRED**

**Error Details:** {str(e)}

**Possible Solutions:**
- Ensure your CSV file is properly formatted
- Check that the file isn't corrupted
- Try uploading a different file
- Verify the file encoding is supported

**Need Help?** Please contact support with the error details above.
        """
        traceback.print_exc()
        processing_msg.content = error_msg
        await processing_msg.update()
