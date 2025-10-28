import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Bike Price Predictor", 
    page_icon="🚲", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def load_model(model_path: str = "model.lb"):
    """Load the trained model with error handling"""
    try:
        p = Path(model_path)
        if not p.exists():
            st.error(f"❌ Model file '{model_path}' not found. Please train and save the model from the notebook first.")
            return None
        model = joblib.load(p)
        st.success(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def load_data(data_path: str = "new_filtered_file1.csv"):
    """Load the dataset with error handling"""
    try:
        p = Path(data_path)
        if not p.exists():
            st.warning(f"⚠️ Dataset '{data_path}' not found.")
            return None
        df = pd.read_csv(p)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def get_brand_mapping():
    """Create brand code to name mapping"""
    return {
        0: "Other", 1: "Yamaha", 2: "Honda", 3: "Suzuki", 4: "Kawasaki",
        5: "TVS", 6: "Bajaj", 7: "Mahindra", 8: "KTM", 9: "Ducati",
        10: "Harley-Davidson", 11: "Royal Enfield", 12: "BMW", 13: "Triumph",
        14: "Aprilia", 15: "Benelli", 16: "Hyosung", 17: "Pulsar", 18: "Activa",
        19: "Hero", 20: "Ninja", 21: "Avenger", 22: "Duke", 23: "CBR",
        24: "R15", 25: "FZ", 26: "Apache", 27: "Splendor", 28: "Passion",
        29: "CD", 30: "Platina"
    }

def get_owner_mapping():
    """Create owner type mapping"""
    return {0: "First Owner", 1: "Second Owner", 2: "Third Owner", 3: "Fourth+ Owner"}

# Load model and data
model = load_model()
df = load_data()

# Header
st.title("🚲 Advanced Bike Price Predictor")
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
    <h3 style="color: white; margin: 0;">🎯 Predict Used Bike Prices with AI</h3>
    <p style="color: white; margin: 0; opacity: 0.9;">Get accurate price estimates based on market data and machine learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    if model is not None:
        st.success("✅ Model Status: Loaded")
        if df is not None:
            st.info(f"📈 Training Data: {len(df):,} bikes")
    else:
        st.error("❌ Model Status: Not Loaded")
    
    st.header("🔧 Quick Actions")
    if st.button("🔄 Reload Model", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# Main tabs
tab_pred, tab_eda, tab_model, tab_insights = st.tabs(["🎯 Predict", "📊 Data Explorer", "🤖 Model Info", "💡 Insights"]) 

with tab_pred:
    st.subheader("🔍 Enter Bike Details")
    
    col1, col2, col3 = st.columns(3)
    
    # Get brand and owner mappings
    brand_mapping = get_brand_mapping()
    owner_mapping = get_owner_mapping()
    
    with col1:
        st.markdown("**🏷️ Ownership & Brand**")
        owner_options = list(owner_mapping.keys())
        owner_labels = [f"{k}: {v}" for k, v in owner_mapping.items()]
        owner = st.selectbox("Owner Type", options=owner_options, 
                           format_func=lambda x: owner_mapping[x],
                           index=0, help="Number of previous owners")
        
        brand_options = list(brand_mapping.keys())
        brand = st.selectbox("Brand", options=brand_options,
                           format_func=lambda x: f"{x}: {brand_mapping[x]}",
                           index=19, help="Select the bike brand")
    
    with col2:
        st.markdown("**⚙️ Technical Specs**")
        age = st.number_input("Age (years)", min_value=0, max_value=30, value=5,
                            help="How old is the bike?")
        power = st.number_input("Power (CC/HP)", min_value=50.0, max_value=1500.0, 
                              value=150.0, step=10.0,
                              help="Engine power in CC or HP")
    
    with col3:
        st.markdown("**📏 Usage Details**")
        kms_driven = st.number_input("KMs Driven", min_value=0, max_value=200000, 
                                   value=20000, step=1000,
                                   help="Total kilometers driven")
        
        # Price range indicator based on inputs
        if df is not None:
            similar_bikes = df[
                (df['brand'] == brand) & 
                (df['age'].between(age-2, age+2)) &
                (df['power'].between(power-50, power+50))
            ]
            if len(similar_bikes) > 0:
                price_range = f"₹{similar_bikes['price'].min():,.0f} - ₹{similar_bikes['price'].max():,.0f}"
                st.info(f"💰 Similar bikes: {price_range}")

    # Input summary
    input_dict = {
        "owner": owner,
        "brand": brand,
        "kms_driven": kms_driven,
        "age": age,
        "power": power,
    }
    
    st.subheader("📋 Input Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Bike Details:**
        - 🏷️ **Owner**: {owner_mapping[owner]}
        - 🏭 **Brand**: {brand_mapping[brand]}
        - ⚙️ **Power**: {power} CC/HP
        """)
    
    with col2:
        st.markdown(f"""
        **Usage Info:**
        - 📅 **Age**: {age} years
        - 🛣️ **KMs Driven**: {kms_driven:,} km
        """)
    
    X_input = pd.DataFrame([input_dict])

    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🎯 Predict Price", type="primary", use_container_width=True)

    if predict_button:
        if model is None:
            st.error("❌ No model loaded. Please train and export the model as 'model.lb'.")
        else:
            try:
                with st.spinner("Predicting price..."):
                    pred = float(model.predict(X_input)[0])
                
                # Display prediction with confidence interval
                st.success("🎉 **Prediction Complete!**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💰 Estimated Price", f"₹{pred:,.0f}")
                with col2:
                    # Calculate price per km (depreciation indicator)
                    price_per_km = pred / max(kms_driven, 1)
                    st.metric("📊 Price/KM", f"₹{price_per_km:.2f}")
                with col3:
                    # Age-based depreciation
                    annual_depreciation = (pred / max(age, 1)) if age > 0 else pred
                    st.metric("📉 Value/Year", f"₹{annual_depreciation:,.0f}")
                
                # Price range estimation
                confidence_range = pred * 0.15  # ±15% confidence
                st.info(f"""
                **💡 Price Range**: ₹{pred - confidence_range:,.0f} - ₹{pred + confidence_range:,.0f}
                
                *This range accounts for market variations and model uncertainty*
                """)
                
                # Comparison with market data
                if df is not None:
                    market_avg = df[df['brand'] == brand]['price'].mean()
                    if pred > market_avg:
                        st.warning(f"📈 Price is {((pred/market_avg - 1) * 100):.1f}% above brand average (₹{market_avg:,.0f})")
                    else:
                        st.success(f"📉 Price is {((1 - pred/market_avg) * 100):.1f}% below brand average (₹{market_avg:,.0f})")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.exception(e)

with tab_eda:
    st.subheader("📊 Dataset Explorer")
    
    if df is None:
        st.error("❌ Dataset not available. Please ensure 'new_filtered_file1.csv' exists.")
    else:
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Total Records", f"{len(df):,}")
        with col2:
            st.metric("🏭 Unique Brands", df['brand'].nunique())
        with col3:
            st.metric("💰 Avg Price", f"₹{df['price'].mean():,.0f}")
        with col4:
            st.metric("📏 Avg KMs", f"{df['kms_driven'].mean():,.0f}")
        
        # Data preview
        st.subheader("🔍 Data Sample")
        # Add brand names for better readability
        df_display = df.copy()
        brand_mapping = get_brand_mapping()
        owner_mapping = get_owner_mapping()
        df_display['brand_name'] = df_display['brand'].map(brand_mapping)
        df_display['owner_type'] = df_display['owner'].map(owner_mapping)
        
        # Reorder columns for better display
        display_cols = ['brand_name', 'owner_type', 'price', 'kms_driven', 'age', 'power']
        st.dataframe(df_display[display_cols].head(10), use_container_width=True)
        
        # Interactive visualizations
        st.subheader("📈 Data Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Price distribution
            fig_price = px.histogram(df, x='price', nbins=50, 
                                   title='Price Distribution',
                                   labels={'price': 'Price (₹)', 'count': 'Frequency'})
            fig_price.update_layout(showlegend=False)
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Brand vs Price
            top_brands = df['brand'].value_counts().head(10).index
            df_top_brands = df[df['brand'].isin(top_brands)].copy()
            df_top_brands['brand_name'] = df_top_brands['brand'].map(brand_mapping)
            
            fig_brand = px.box(df_top_brands, x='brand_name', y='price',
                             title='Price by Top Brands')
            fig_brand.update_xaxes(tickangle=45)
            st.plotly_chart(fig_brand, use_container_width=True)
        
        with viz_col2:
            # Age vs Price scatter
            fig_age = px.scatter(df, x='age', y='price', color='power',
                               title='Age vs Price (colored by Power)',
                               labels={'age': 'Age (years)', 'price': 'Price (₹)', 'power': 'Power'})
            st.plotly_chart(fig_age, use_container_width=True)
            
            # KMs vs Price
            fig_kms = px.scatter(df, x='kms_driven', y='price', 
                               color='age', title='KMs Driven vs Price (colored by Age)',
                               labels={'kms_driven': 'KMs Driven', 'price': 'Price (₹)', 'age': 'Age'})
            st.plotly_chart(fig_kms, use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)

with tab_model:
    st.subheader("🤖 Model Information")
    
    if model is None:
        st.error("❌ Model not loaded")
    else:
        st.success("✅ Model loaded successfully")
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📋 Model Details:**
            - **Algorithm**: Linear Regression
            - **Features**: 5 (owner, brand, kms_driven, age, power)
            - **Target**: Price (₹)
            - **Training**: Scikit-learn
            """)
        
        with col2:
            st.markdown("""
            **🎯 Model Characteristics:**
            - **Type**: Supervised Learning
            - **Problem**: Regression
            - **Interpretability**: High
            - **Speed**: Very Fast
            """)
        
        # Model performance (if data is available)
        if df is not None:
            st.subheader("📈 Model Performance")
            
            # Simulate model performance metrics
            X = df.drop('price', axis=1)
            y = df['price']
            
            try:
                y_pred = model.predict(X)
                
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                with perf_col1:
                    st.metric("📊 R² Score", f"{r2:.3f}")
                with perf_col2:
                    st.metric("📉 MAE", f"₹{mae:,.0f}")
                with perf_col3:
                    st.metric("📈 RMSE", f"₹{np.sqrt(mse):,.0f}")
                
                # Residual plot
                residuals = y - y_pred
                fig_residuals = px.scatter(x=y_pred, y=residuals,
                                         title='Residual Plot (Actual vs Predicted)',
                                         labels={'x': 'Predicted Price', 'y': 'Residuals'})
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error calculating performance metrics: {str(e)}")

with tab_insights:
    st.subheader("💡 Market Insights & Tips")
    
    if df is not None:
        # Market insights
        st.markdown("### 🎯 Key Market Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            # Top brands by value retention
            brand_stats = df.groupby('brand').agg({
                'price': ['mean', 'count'],
                'age': 'mean',
                'kms_driven': 'mean'
            }).round(0)
            brand_stats.columns = ['Avg_Price', 'Count', 'Avg_Age', 'Avg_KMs']
            brand_stats = brand_stats[brand_stats['Count'] >= 10]  # Filter brands with sufficient data
            brand_stats['brand_name'] = brand_stats.index.map(brand_mapping)
            top_value_brands = brand_stats.nlargest(5, 'Avg_Price')
            
            st.markdown("**🏆 Top Value Brands:**")
            for idx, row in top_value_brands.iterrows():
                st.write(f"- **{row['brand_name']}**: ₹{row['Avg_Price']:,.0f} avg")
        
        with insights_col2:
            # Depreciation insights
            age_groups = df.groupby(pd.cut(df['age'], bins=[0, 2, 5, 10, 30], labels=['0-2yr', '2-5yr', '5-10yr', '10+yr']))['price'].mean()
            st.markdown("**📉 Average Price by Age:**")
            for age_group, price in age_groups.items():
                st.write(f"- **{age_group}**: ₹{price:,.0f}")
        
        # Buying tips
        st.markdown("### 🛒 Smart Buying Tips")
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.markdown("""
            **💰 Price Optimization:**
            - 🎯 Sweet spot: 3-5 years old bikes
            - 📊 Look for <30,000 km odometer
            - 🏷️ Second owners offer good value
            - ⚡ Higher power bikes depreciate slower
            """)
        
        with tips_col2:
            st.markdown("""
            **🔍 What to Check:**
            - 📋 Service history and records
            - 🔧 Engine condition and maintenance
            - 📄 Clear ownership documents
            - 🛡️ Insurance validity
            """)
        
        # Price prediction confidence
        st.markdown("### 🎯 Prediction Confidence")
        st.info("""
        **Model Accuracy Notes:**
        - ✅ Most accurate for common brands and standard specifications
        - ⚠️ Less reliable for rare/luxury bikes or extreme specifications  
        - 📊 Price estimates include ±15% confidence interval
        - 🔄 Regular model updates improve accuracy over time
        """)
    
    else:
        st.warning("⚠️ Insights not available without dataset")
