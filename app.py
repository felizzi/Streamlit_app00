import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Data Entry App",
    page_icon="📊",
    layout="wide"
)

# App title
st.title("📊 Data Entry Application")
st.markdown("Enter your data and watch it build into a dataframe!")

# Initialize session state for the dataframe
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Name', 'Age', 'City', 'Score'])

# Create two columns for the input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    name = st.text_input("👤 Name", placeholder="Enter full name")
    age = st.number_input("🎂 Age", min_value=0, max_value=120, value=25, step=1)

with col2:
    st.subheader("Additional Details")
    city = st.text_input("🏙️ City", placeholder="Enter city name")
    score = st.number_input("⭐ Score", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

# Form submission
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn1:
    if st.button("➕ Add Data", type="primary", use_container_width=True):
        if name.strip() and city.strip():  # Basic validation
            # Create new row
            new_row = pd.DataFrame({
                'Name': [name.strip()],
                'Age': [age],
                'City': [city.strip()],
                'Score': [score]
            })
            
            # Add to existing dataframe
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            
            st.success(f"✅ Data added! Total entries: {len(st.session_state.df)}")
            st.balloons()  # Fun animation!
            
        else:
            st.error("❌ Please fill in at least Name and City fields")

with col_btn2:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.df = pd.DataFrame(columns=['Name', 'Age', 'City', 'Score'])
        st.success("🧹 All data cleared!")

# Display the dataframe and statistics
if not st.session_state.df.empty:
    st.markdown("---")
    st.subheader("📈 Your Data")
    
    # Display dataframe with formatting
    st.dataframe(
        st.session_state.df,
        use_container_width=True,
        hide_index=True
    )
    
    # Statistics in columns
    st.subheader("📊 Summary Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Entries", len(st.session_state.df))
    
    with stat_col2:
        avg_age = st.session_state.df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f}")
    
    with stat_col3:
        avg_score = st.session_state.df['Score'].mean()
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with stat_col4:
        unique_cities = st.session_state.df['City'].nunique()
        st.metric("Unique Cities", unique_cities)
    
    # Data visualization
    if len(st.session_state.df) > 1:
        st.subheader("📊 Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.bar_chart(st.session_state.df.set_index('Name')['Score'])
            st.caption("Scores by Person")
        
        with viz_col2:
            st.bar_chart(st.session_state.df.set_index('Name')['Age'])
            st.caption("Ages by Person")
    
    # Download functionality
    st.markdown("---")
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="data_entry.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_download2:
        json_data = st.session_state.df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name="data_entry.json",
            mime="application/json",
            use_container_width=True
        )

else:
    st.info("📝 No data entered yet. Use the form above to add your first entry!")
    
# Sidebar with instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Fill out the form** with your information
    2. **Click 'Add Data'** to store it
    3. **View your data** in the table below
    4. **Download** your data when ready
    5. **Clear all** to start fresh
    """)
    
    st.header("🎯 Features")
    st.markdown("""
    - ✅ Real-time data entry
    - 📊 Automatic statistics
    - 📈 Data visualizations  
    - 📥 CSV & JSON export
    - 🧹 Easy data clearing
    """)
    
    if not st.session_state.df.empty:
        st.header("🔍 Data Preview")
        st.write(f"**{len(st.session_state.df)}** entries stored")
        st.write("Latest entry:")
        st.write(st.session_state.df.iloc[-1].to_dict())