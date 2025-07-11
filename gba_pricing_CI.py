import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

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
    .confidence-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained PKL model
@st.cache_resource
def load_pkl_model():
    """Load your trained PKL model"""
    try:
        # Try to load your PKL model (adjust filename as needed)
        model_files = [
            'xgboost_gba_model.pkl',
            'corrected_xgboost_model.pkl', 
            'ordered_xgboost_model.pkl',
            'your_xgboost_model.pkl',
            'gba_pricing_model.pkl'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                return model_data, model_file
        
        return None, None
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model at startup
model_data, model_filename = load_pkl_model()

# App title and header
st.title("💊 G-BA Pharmaceutical Pricing Predictor")
st.markdown("**Predict new product pricing based on G-BA assessment criteria and comparator analysis**")

# Model status
if model_data:
    st.success(f"🤖 ML Model loaded: {model_filename}")
    if 'model_type' in model_data:
        st.info(f"Model: {model_data.get('model_type', 'Unknown')} | R² Score: {model_data.get('r2_score', 'N/A')}")
else:
    st.warning("⚠️ No PKL model found. Using rule-based prediction algorithm.")

# Initialize session state for storing predictions
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = pd.DataFrame(columns=[
        'Product_Name', 'Comparator_Price_Mean', 'QoL_Assessment', 'Mortality_Assessment', 
        'Morbidity_Assessment', 'Safety_Assessment', 'Study_Design_Assessment', 
        'Overall_GBA_Assessment', 'Additional_Benefit_Probability', 'Predicted_Price',
        'Lower_CI', 'Upper_CI', 'Prediction_Method'
    ])

# Function to map string to numeric value for model input
def map_benefit_probability(benefit_string):
    """Map benefit probability string to numeric value for model"""
    mapping = {
        "No additional benefit": 0,
        "Hint for additional benefit": 1,
        "Minor additional benefit": 2,
        "Considerable benefit": 3,
        "Major additional benefit": 4
    }
    return mapping.get(benefit_string, 2)  # Default to "Minor" if not found

# Confidence interval calculation
def calculate_confidence_intervals(prediction, method='simple', confidence_level=0.95):
    """Calculate confidence intervals for predictions"""
    
    if method == 'simple':
        # Simple method using estimated uncertainty
        # Adjust this based on your model's validation performance
        estimated_std_error = 0.12  # 12% uncertainty (adjust based on your model performance)
        
        alpha = 1 - confidence_level
        z_score = 1.96 if confidence_level == 0.95 else 2.576
        
        margin_of_error = z_score * estimated_std_error * prediction
        lower_bound = prediction - margin_of_error
        upper_bound = prediction + margin_of_error
        
        return {
            'lower_bound': max(0, lower_bound),  # Ensure non-negative
            'upper_bound': upper_bound,
            'margin_of_error': margin_of_error,
            'confidence_level': confidence_level
        }
    
    return None

# Enhanced prediction function that uses PKL model if available
def make_prediction_with_model(comparator_price, qol, mortality, morbidity, safety, study_design, overall_gba, benefit_prob_str):
    """Make prediction using PKL model if available, fallback to rule-based"""
    
    # Convert benefit probability string to numeric
    benefit_prob_numeric = map_benefit_probability(benefit_prob_str)
    
    if model_data and 'model' in model_data:
        try:
            # Use PKL model for prediction
            
            # Create input in the correct order for your model
            feature_order = model_data.get('feature_order', [
                'addit benefit probabl',
                'g ba best assess studi design', 
                'log price act',
                'GBA_mortality_lin',
                'GBA_morbidity_lin', 
                'GBA_qol_lin',
                'GBA_safety_lin',
                'GBA_assessment'
            ])
            
            # Map inputs to model features
            input_mapping = {
                'addit benefit probabl': benefit_prob_numeric,
                'g ba best assess studi design': study_design,
                'log price act': np.log(comparator_price),
                'GBA_mortality_lin': mortality,
                'GBA_morbidity_lin': morbidity,
                'GBA_qol_lin': qol,
                'GBA_safety_lin': safety,
                'GBA_assessment': overall_gba
            }
            
            # Create input array in correct order
            input_values = []
            for feature in feature_order:
                if feature in input_mapping:
                    input_values.append(input_mapping[feature])
                else:
                    input_values.append(0)  # Default value
            
            input_data = np.array([input_values])
            
            # Make prediction (assumes log price premium)
            log_premium = model_data['model'].predict(input_data)[0]
            predicted_price = comparator_price * np.exp(log_premium)
            
            return predicted_price, "ML Model (XGBoost)"
            
        except Exception as e:
            st.warning(f"ML model failed: {e}. Using rule-based prediction.")
            # Fall back to rule-based prediction
            return calculate_predicted_price_rule_based(
                comparator_price, qol, mortality, morbidity, safety, 
                study_design, overall_gba, benefit_prob_numeric
            ), "Rule-based (Fallback)"
    
    else:
        # Use rule-based prediction
        return calculate_predicted_price_rule_based(
            comparator_price, qol, mortality, morbidity, safety, 
            study_design, overall_gba, benefit_prob_numeric
        ), "Rule-based Algorithm"

# Original rule-based prediction function (modified to accept numeric benefit prob)
def calculate_predicted_price_rule_based(comparator_price, qol, mortality, morbidity, safety, study_design, overall_gba, benefit_prob):
    """Original rule-based prediction algorithm"""
    # Convert scores to inverted scale (1=best becomes 6, 6=worst becomes 1)
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
    
    # Additional benefit multiplier (15% weight) - convert 0-4 scale to percentage-like
    benefit_pct = (benefit_prob / 4) * 100  # Convert 0-4 to 0-100
    benefit_multiplier = 1 + (benefit_pct / 100) * 0.3  # Range: 1.0 to 1.3
    
    # Market position adjustment (25% weight) - using inverted overall score
    market_adjustment = 1 + (overall_gba_inv - 3.5) * 0.08  # Range: 0.8 to 1.2
    
    # Calculate predicted price
    predicted_price = comparator_price * gba_multiplier * benefit_multiplier * market_adjustment
    
    return round(predicted_price, 2)

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
    """)
    
    if model_data:
        st.success("🤖 **Machine Learning Model Active**")
        st.markdown("- XGBoost with regularization")
        st.markdown("- Trained on historical G-BA data")
        st.markdown("- Includes confidence intervals")
    else:
        st.info("📊 **Rule-based Algorithm**")
        st.markdown("- Weighted scoring system")
        st.markdown("- G-BA assessment scores (60% weight)")
        st.markdown("- Comparator pricing (25% weight)")
        st.markdown("- Additional benefit probability (15% weight)")
    
    st.markdown("---")
    st.header("💡 How It Works")
    st.markdown("""
    1. **Enter product details** and assessment scores
    2. **Review the prediction** with confidence intervals
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
        
        # Changed to dropdown instead of slider
        additional_benefit_prob = st.selectbox(
            "🎲 Additional Benefit Probability",
            options=[
                "No additional benefit",
                "Hint for additional benefit", 
                "Minor additional benefit",
                "Considerable benefit",
                "Major additional benefit"
            ],
            index=2,
            help="Select the level of additional benefit expected"
        )

with col2:
    st.header("🔮 Pricing Prediction")
    
    # Calculate prediction with confidence intervals
    if comparator_price > 0:
        predicted_price, method = make_prediction_with_model(
            comparator_price, qol_score, mortality_score, morbidity_score, 
            safety_score, study_design_score, overall_gba_score, additional_benefit_prob
        )
        
        # Calculate confidence intervals
        ci_result = calculate_confidence_intervals(predicted_price)
        
        # Display prediction with styling
        st.markdown(f"""
        <div class="prediction-box">
            <h3>💰 Predicted Price</h3>
            <h2 style="color: #1f77b4;">€{predicted_price:,.2f}</h2>
            <p>Price premium vs. comparators: <strong>{((predicted_price/comparator_price - 1) * 100):+.1f}%</strong></p>
            <p><em>Method: {method}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display confidence intervals
        if ci_result:
            st.markdown(f"""
            <div class="confidence-box">
                <h4>📊 95% Confidence Interval</h4>
                <p><strong>Range: €{ci_result['lower_bound']:,.0f} - €{ci_result['upper_bound']:,.0f}</strong></p>
                <p>Margin of Error: ±€{ci_result['margin_of_error']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence assessment
            ci_width_pct = (ci_result['margin_of_error'] * 2 / predicted_price) * 100
            if ci_width_pct < 20:
                confidence_level = "🟢 High Confidence"
            elif ci_width_pct < 35:
                confidence_level = "🟡 Moderate Confidence"
            else:
                confidence_level = "🔴 Lower Confidence"
            
            st.info(f"**{confidence_level}** (±{ci_width_pct:.1f}%)")
        
        # Price breakdown
        st.subheader("📊 Price Factors")
        
        # Calculate individual contributions (using original scores for display)
        gba_composite = (qol_score + mortality_score + morbidity_score + safety_score + study_design_score + overall_gba_score) / 6
        benefit_numeric = map_benefit_probability(additional_benefit_prob)
        
        factors_df = pd.DataFrame({
            'Factor': ['G-BA Composite Score', 'Additional Benefit Level', 'Comparator Price'],
            'Value': [f"{gba_composite:.1f}/6 (lower=better)", additional_benefit_prob, f"€{comparator_price:,.0f}"],
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
            predicted_price, method = make_prediction_with_model(
                comparator_price, qol_score, mortality_score, morbidity_score, 
                safety_score, study_design_score, overall_gba_score, additional_benefit_prob
            )
            
            # Calculate confidence intervals for saving
            ci_result = calculate_confidence_intervals(predicted_price)
            
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
                'Predicted_Price': [predicted_price],
                'Lower_CI': [ci_result['lower_bound'] if ci_result else 0],
                'Upper_CI': [ci_result['upper_bound'] if ci_result else 0],
                'Prediction_Method': [method]
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
            'Overall_GBA_Assessment', 'Additional_Benefit_Probability', 'Predicted_Price',
            'Lower_CI', 'Upper_CI', 'Prediction_Method'
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
        st.subheader("💰 Price Distribution with CI")
        # Create chart data with confidence intervals
        chart_data = st.session_state.predictions_df[['Product_Name', 'Predicted_Price', 'Lower_CI', 'Upper_CI']].set_index('Product_Name')
        st.bar_chart(chart_data)
    
    with col_analytics2:
        st.subheader("🎯 Prediction Methods")
        method_counts = st.session_state.predictions_df['Prediction_Method'].value_counts()
        st.bar_chart(method_counts)
    
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
        if 'Lower_CI' in st.session_state.predictions_df.columns:
            avg_uncertainty = ((st.session_state.predictions_df['Upper_CI'] - st.session_state.predictions_df['Lower_CI']) / st.session_state.predictions_df['Predicted_Price']).mean() * 100
            st.metric("Average Uncertainty", f"±{avg_uncertainty:.1f}%")
        else:
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
            file_name="gba_pricing_predictions_with_ci.csv",
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
                'Model_Used': st.session_state.predictions_df['Prediction_Method'].iloc[0] if 'Prediction_Method' in st.session_state.predictions_df.columns else 'Unknown',
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

# Footer with model information
st.markdown("---")
if model_data:
    st.info("🤖 **ML Model Active**: Predictions use your trained XGBoost model with confidence intervals. Fallback to rule-based algorithm if model fails.")
else:
    st.info("📊 **Rule-based Mode**: No PKL model detected. Using weighted scoring algorithm. Place your trained model file in the app directory to enable ML predictions.")