import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="G-BA Pricing Predictor",
    page_icon="💊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# App title and header
st.title("💊 G-BA Pharmaceutical Pricing Predictor")
st.markdown("**Predict new product pricing based on G-BA assessment criteria and comparator analysis**")

# Initialize session state for storing predictions
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = pd.DataFrame(columns=[
        'Product_Name', 'Comparator_Price_Mean', 'QoL_Assessment', 'Mortality_Assessment', 
        'Morbidity_Assessment', 'Safety_Assessment', 'Study_Design_Assessment', 
        'Overall_GBA_Assessment', 'Additional_Benefit_Probability', 'Predicted_Price'
    ])

# Sidebar with information and comments
with st.sidebar:
    st.header("📋 About This Tool")
    st.markdown("""
    This application predicts pharmaceutical product pricing based on:
    
    **🎯 G-BA Assessment Criteria:**
    - Quality of Life (QoL) impact
    - Mortality benefits
    - Morbidity improvements
    - Safety profile
    - Study design quality
    - Overall G-BA rating
    
    **📊 Market Analysis:**
    - Comparator pricing data
    - Additional benefit probability
    
    **🔮 Prediction Model:**
    The app uses a weighted scoring algorithm that considers:
    - G-BA assessment scores (60% weight)
    - Comparator pricing (25% weight)
    - Additional benefit probability (15% weight)
    """)
    
    st.markdown("---")
    st.header("💡 How It Works")
    st.markdown("""
    1. **Enter product details** and assessment scores
    2. **Review the prediction** in real-time
    3. **Save predictions** to build a database
    4. **Analyze trends** across multiple products
    """)
    
    st.markdown("---")
    st.header("📈 Market Context")
    st.info("Higher G-BA scores (lower numbers) typically correlate with premium pricing opportunities, especially for products addressing unmet medical needs.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Product Assessment Input")
    
    # Product identification
    product_name = st.text_input("🏷️ Product Name", placeholder="Enter product name")
    
    # Create input sections
    st.subheader("💰 Market Comparators")
    comparator_price = st.number_input(
        "Price of Comparators (Mean) €", 
        min_value=0.0, 
        value=1000.0, 
        step=50.0,
        help="Average price of comparable products in the therapeutic area"
    )
    
    st.subheader("🏥 G-BA Assessment Scores")
    st.markdown("*Rate each criterion from 1 (Excellent) to 6 (Poor)*")
    
    # G-BA assessment inputs with descriptions
    col_a, col_b = st.columns(2)
    
    with col_a:
        qol_score = st.selectbox(
            "🌟 Quality of Life Assessment", 
            options=[1, 2, 3, 4, 5, 6],
            index=2,
            help="Impact on patient quality of life (1=Excellent, 6=Poor)"
        )
        
        mortality_score = st.selectbox(
            "⚕️ Mortality Assessment", 
            options=[1, 2, 3, 4, 5, 6],
            index=2,
            help="Effect on patient survival (1=Excellent, 6=Poor)"
        )
        
        morbidity_score = st.selectbox(
            "🩺 Morbidity Assessment", 
            options=[1, 2, 3, 4, 5, 6],
            index=2,
            help="Impact on disease progression/symptoms (1=Excellent, 6=Poor)"
        )
        
        safety_score = st.selectbox(
            "🛡️ Safety Assessment", 
            options=[1, 2, 3, 4, 5, 6],
            index=1,
            help="Safety profile and adverse events (1=Excellent, 6=Poor)"
        )
    
    with col_b:
        study_design_score = st.selectbox(
            "🔬 Study Design Assessment", 
            options=[1, 2, 3, 4, 5, 6],
            index=1,
            help="Quality of clinical trial design (1=Excellent, 6=Poor)"
        )
        
        overall_gba_score = st.selectbox(
            "📊 Overall G-BA Assessment", 
            options=[1, 2, 3, 4, 5, 6],
            index=2,
            help="G-BA's overall rating (1=Excellent, 6=Poor)"
        )
        
        additional_benefit_prob = st.slider(
            "🎲 Additional Benefit Probability (%)", 
            min_value=0, 
            max_value=100, 
            value=50,
            help="Probability of demonstrating additional benefit"
        )

with col2:
    st.header("🔮 Pricing Prediction")
    
    # Pricing prediction algorithm
    def calculate_predicted_price(comparator_price, qol, mortality, morbidity, safety, study_design, overall_gba, benefit_prob):
        # Convert scores to inverted scale (1=best becomes 6, 6=worst becomes 1)
        # This way we can use the same logic but with inverted scores
        qol_inv = 7 - qol
        mortality_inv = 7 - mortality
        morbidity_inv = 7 - morbidity
        safety_inv = 7 - safety
        study_design_inv = 7 - study_design
        overall_gba_inv = 7 - overall_gba
        
        # Weighted scoring algorithm using inverted scores
        gba_composite_score = (qol_inv + mortality_inv + morbidity_inv + safety_inv + study_design_inv + overall_gba_inv) / 6
        
        # Base multiplier based on G-BA scores (60% weight)
        gba_multiplier = 0.5 + (gba_composite_score - 1) * 0.2  # Range: 0.5 to 1.5
        
        # Additional benefit multiplier (15% weight)
        benefit_multiplier = 1 + (benefit_prob / 100) * 0.3  # Range: 1.0 to 1.3
        
        # Market position adjustment (25% weight) - using inverted overall score
        market_adjustment = 1 + (overall_gba_inv - 3.5) * 0.08  # Range: 0.8 to 1.2
        
        # Calculate predicted price
        predicted_price = comparator_price * gba_multiplier * benefit_multiplier * market_adjustment
        
        return round(predicted_price, 2)
    
    # Calculate prediction
    if comparator_price > 0:
        predicted_price = calculate_predicted_price(
            comparator_price, qol_score, mortality_score, morbidity_score, 
            safety_score, study_design_score, overall_gba_score, additional_benefit_prob
        )
        
        # Display prediction with styling
        st.markdown(f"""
        <div class="prediction-box">
            <h3>💰 Predicted Price</h3>
            <h2 style="color: #1f77b4;">€{predicted_price:,.2f}</h2>
            <p>Price premium vs. comparators: <strong>{((predicted_price/comparator_price - 1) * 100):+.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Price breakdown
        st.subheader("📊 Price Factors")
        
        # Calculate individual contributions (using original scores for display)
        gba_composite = (qol_score + mortality_score + morbidity_score + safety_score + study_design_score + overall_gba_score) / 6
        
        factors_df = pd.DataFrame({
            'Factor': ['G-BA Composite Score', 'Additional Benefit Prob.', 'Comparator Price'],
            'Value': [f"{gba_composite:.1f}/6 (lower=better)", f"{additional_benefit_prob}%", f"€{comparator_price:,.0f}"],
            'Impact': ['High', 'Medium', 'Base']
        })
        
        st.dataframe(factors_df, hide_index=True, use_container_width=True)
        
        # G-BA scores visualization using Streamlit charts
        st.subheader("🎯 G-BA Score Profile")
        
        # Create a dataframe for the scores
        scores_df = pd.DataFrame({
            'Criterion': ['QoL', 'Mortality', 'Morbidity', 'Safety', 'Study Design', 'Overall'],
            'Score': [qol_score, mortality_score, morbidity_score, safety_score, study_design_score, overall_gba_score],
            'Inverted_Score': [7-qol_score, 7-mortality_score, 7-morbidity_score, 7-safety_score, 7-study_design_score, 7-overall_gba_score]
        })
        
        # Display as bar chart (inverted so higher bars = better scores)
        st.bar_chart(scores_df.set_index('Criterion')['Inverted_Score'], height=300)
        st.caption("Higher bars = Better G-BA scores (lower numerical values)")
        
        # Display the actual scores in a table
        display_scores = scores_df[['Criterion', 'Score']].copy()
        display_scores['Rating'] = display_scores['Score'].map({
            1: '1 - Excellent', 2: '2 - Very Good', 3: '3 - Good', 
            4: '4 - Moderate', 5: '5 - Poor', 6: '6 - Very Poor'
        })
        st.dataframe(display_scores[['Criterion', 'Rating']], hide_index=True, use_container_width=True)

# Save prediction functionality
st.markdown("---")
col_save1, col_save2, col_save3 = st.columns([1, 1, 1])

with col_save1:
    if st.button("💾 Save Prediction", type="primary", use_container_width=True):
        if product_name.strip() and comparator_price > 0:
            predicted_price = calculate_predicted_price(
                comparator_price, qol_score, mortality_score, morbidity_score, 
                safety_score, study_design_score, overall_gba_score, additional_benefit_prob
            )
            
            new_row = pd.DataFrame({
                'Product_Name': [product_name.strip()],
                'Comparator_Price_Mean': [comparator_price],
                'QoL_Assessment': [qol_score],
                'Mortality_Assessment': [mortality_score],
                'Morbidity_Assessment': [morbidity_score],
                'Safety_Assessment': [safety_score],
                'Study_Design_Assessment': [study_design_score],
                'Overall_GBA_Assessment': [overall_gba_score],
                'Additional_Benefit_Probability': [additional_benefit_prob],
                'Predicted_Price': [predicted_price]
            })
            
            st.session_state.predictions_df = pd.concat([st.session_state.predictions_df, new_row], ignore_index=True)
            st.success(f"✅ Prediction saved for {product_name}!")
            st.balloons()
        else:
            st.error("❌ Please enter a product name and valid comparator price")

with col_save2:
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.predictions_df = pd.DataFrame(columns=[
            'Product_Name', 'Comparator_Price_Mean', 'QoL_Assessment', 'Mortality_Assessment', 
            'Morbidity_Assessment', 'Safety_Assessment', 'Study_Design_Assessment', 
            'Overall_GBA_Assessment', 'Additional_Benefit_Probability', 'Predicted_Price'
        ])
        st.success("🧹 All data cleared!")

# Display saved predictions
if not st.session_state.predictions_df.empty:
    st.header("📈 Saved Predictions & Analysis")
    
    # Display the data table
    st.subheader("📋 Predictions Database")
    st.dataframe(st.session_state.predictions_df, hide_index=True, use_container_width=True)
    
    # Analytics section using Streamlit built-in charts
    col_analytics1, col_analytics2 = st.columns(2)
    
    with col_analytics1:
        st.subheader("💰 Price Distribution")
        st.histogram_chart(st.session_state.predictions_df['Predicted_Price'], bins=10)
    
    with col_analytics2:
        st.subheader("🎯 G-BA Scores vs Price")
        chart_data = st.session_state.predictions_df[['Overall_GBA_Assessment', 'Predicted_Price']].copy()
        st.scatter_chart(chart_data.set_index('Overall_GBA_Assessment'))
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        avg_price = st.session_state.predictions_df['Predicted_Price'].mean()
        st.metric("Average Predicted Price", f"€{avg_price:,.0f}")
    
    with stats_col2:
        avg_gba = st.session_state.predictions_df['Overall_GBA_Assessment'].mean()
        st.metric("Average G-BA Score", f"{avg_gba:.1f}/6 (lower=better)")
    
    with stats_col3:
        price_range = st.session_state.predictions_df['Predicted_Price'].max() - st.session_state.predictions_df['Predicted_Price'].min()
        st.metric("Price Range", f"€{price_range:,.0f}")
    
    with stats_col4:
        total_products = len(st.session_state.predictions_df)
        st.metric("Total Products", total_products)
    
    # Download functionality
    st.markdown("---")
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        csv = st.session_state.predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="gba_pricing_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_download2:
        # Create summary report
        if len(st.session_state.predictions_df) > 0:
            summary_stats = {
                'Total_Products': len(st.session_state.predictions_df),
                'Average_Predicted_Price': float(st.session_state.predictions_df['Predicted_Price'].mean()),
                'Average_GBA_Score': float(st.session_state.predictions_df['Overall_GBA_Assessment'].mean()),
                'Price_Range': float(price_range),
                'Highest_Priced_Product': st.session_state.predictions_df.loc[st.session_state.predictions_df['Predicted_Price'].idxmax(), 'Product_Name'],
                'Best_GBA_Score': float(st.session_state.predictions_df['Overall_GBA_Assessment'].min())  # Min because 1 is best
            }
            
            summary_json = pd.Series(summary_stats).to_json(indent=2)
            st.download_button(
                label="📥 Download Summary Report",
                data=summary_json,
                file_name="gba_summary_report.json",
                mime="application/json",
                use_container_width=True
            )

else:
    st.info("📝 No predictions saved yet. Enter product details above and click 'Save Prediction' to start building your database.")