import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Hospital Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD AND PROCESS DATA
# ============================================
@st.cache_data
def load_data():
    """Load data dari CSV dan lakukan preprocessing"""
    df = pd.read_csv('data/healthcare_dataset.csv')
    
    # Convert date columns to datetime
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    
    # Calculate Length of Stay (LOS)
    df['LOS'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    
    return df

# Load data
df = load_data()

# ============================================
# SIDEBAR - FILTERS
# ============================================
st.sidebar.header("🔍 Filters")

# Hospital Filter
all_hospitals = ['All Hospitals'] + sorted(df['Hospital'].unique().tolist())
selected_hospitals = st.sidebar.multiselect(
    "Select Hospital(s):",
    options=all_hospitals,
    default=['All Hospitals']
)

# Medical Condition Filter
all_conditions = ['All Conditions'] + sorted(df['Medical Condition'].unique().tolist())
selected_conditions = st.sidebar.multiselect(
    "Select Medical Condition(s):",
    options=all_conditions,
    default=['All Conditions']
)

# Insurance Provider Filter
all_insurance = ['All Insurance'] + sorted(df['Insurance Provider'].unique().tolist())
selected_insurance = st.sidebar.multiselect(
    "Select Insurance Provider(s):",
    options=all_insurance,
    default=['All Insurance']
)

# Admission Type Filter
all_admission = ['All Types'] + sorted(df['Admission Type'].unique().tolist())
selected_admission = st.sidebar.multiselect(
    "Select Admission Type(s):",
    options=all_admission,
    default=['All Types']
)

# ============================================
# APPLY FILTERS
# ============================================
filtered_df = df.copy()

# Filter Hospital
if 'All Hospitals' not in selected_hospitals:
    filtered_df = filtered_df[filtered_df['Hospital'].isin(selected_hospitals)]

# Filter Medical Condition
if 'All Conditions' not in selected_conditions:
    filtered_df = filtered_df[filtered_df['Medical Condition'].isin(selected_conditions)]

# Filter Insurance
if 'All Insurance' not in selected_insurance:
    filtered_df = filtered_df[filtered_df['Insurance Provider'].isin(selected_insurance)]

# Filter Admission Type
if 'All Types' not in selected_admission:
    filtered_df = filtered_df[filtered_df['Admission Type'].isin(selected_admission)]

# ============================================
# MAIN CONTENT - HEADER
# ============================================
st.title("🏥 Hospital Operational Analytics Dashboard")
st.markdown("Analitik Deskriptif Operasional Rumah Sakit")

# Display data range info
st.info(f"📊 Total Patients: {len(filtered_df):,} | Data Range: {df['Date of Admission'].min().date()} to {df['Discharge Date'].max().date()}")

# ============================================
# KEY PERFORMANCE INDICATORS (KPI)
# ============================================
st.subheader("📈 Key Performance Indicators (KPI)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_patients = len(filtered_df)
    st.metric(
        label="Total Patients",
        value=f"{total_patients:,}",
        delta=f"{(total_patients/len(df)*100):.1f}% of total"
    )

with col2:
    avg_billing = filtered_df['Billing Amount'].mean()
    st.metric(
        label="Avg Billing Amount",
        value=f"${avg_billing:,.2f}",
        delta=f"Min: ${filtered_df['Billing Amount'].min():,.0f}"
    )

with col3:
    avg_los = filtered_df['LOS'].mean()
    st.metric(
        label="Avg Length of Stay",
        value=f"{avg_los:.1f} days",
        delta=f"Max: {filtered_df['LOS'].max()} days"
    )

with col4:
    total_hospitals = filtered_df['Hospital'].nunique()
    st.metric(
        label="Total Hospitals",
        value=f"{total_hospitals}",
        delta=f"of {df['Hospital'].nunique()} hospitals"
    )

st.divider()

# ============================================
# ROW 1: VISUALIZATIONS (TOP)
# ============================================
st.subheader("📊 Analysis Overview")

col_a, col_b = st.columns(2)

# VISUALIZATION 1: Distribution of Medical Conditions (Pie Chart)
with col_a:
    condition_counts = filtered_df['Medical Condition'].value_counts().reset_index()
    condition_counts.columns = ['Medical Condition', 'Count']
    
    fig_condition = px.pie(
        condition_counts,
        values='Count',
        names='Medical Condition',
        title='Distribution of Medical Conditions',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_condition.update_traces(textposition='auto', textinfo='percent+label')
    
    st.plotly_chart(fig_condition, use_container_width=True)

# VISUALIZATION 2: Average Billing by Hospital (Bar Chart)
with col_b:
    hospital_billing = filtered_df.groupby('Hospital')['Billing Amount'].mean().reset_index()
    hospital_billing = hospital_billing.sort_values('Billing Amount', ascending=False)
    
    fig_hospital = px.bar(
        hospital_billing,
        x='Hospital',
        y='Billing Amount',
        title='Average Billing Amount by Hospital',
        color='Billing Amount',
        color_continuous_scale='Viridis'
    )
    fig_hospital.update_layout(
        xaxis_tickangle=-45,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_hospital, use_container_width=True)

# ============================================
# ROW 2: VISUALIZATIONS (MIDDLE)
# ============================================
col_c, col_d = st.columns(2)

# VISUALIZATION 3: Age Distribution (Histogram)
with col_c:
    fig_age = px.histogram(
        filtered_df,
        x='Age',
        nbins=20,
        title='Distribution of Patient Age',
        color_discrete_sequence=['#636EFA'],
        marginal='box'
    )
    fig_age.update_layout(hovermode='x unified')
    
    st.plotly_chart(fig_age, use_container_width=True)

# VISUALIZATION 4: Admission Type Distribution (Bar Chart)
with col_d:
    admission_counts = filtered_df['Admission Type'].value_counts().reset_index()
    admission_counts.columns = ['Admission Type', 'Count']
    
    fig_admission = px.bar(
        admission_counts,
        x='Admission Type',
        y='Count',
        title='Distribution of Admission Type',
        color='Admission Type',
        color_discrete_sequence=['#EF553B', '#00CC96', '#AB63FA']
    )
    fig_admission.update_layout(hovermode='x unified')
    
    st.plotly_chart(fig_admission, use_container_width=True)

# ============================================
# ROW 3: VISUALIZATIONS (BOTTOM)
# ============================================
st.subheader("💡 Additional Insights")

col_e, col_f = st.columns(2)

# VISUALIZATION 5: Top Insurance Providers (Bar Chart)
with col_e:
    insurance_counts = filtered_df['Insurance Provider'].value_counts().head(8).reset_index()
    insurance_counts.columns = ['Insurance Provider', 'Count']
    
    fig_insurance = px.bar(
        insurance_counts,
        y='Insurance Provider',
        x='Count',
        title='Top 8 Insurance Providers by Patient Count',
        color='Count',
        color_continuous_scale='Blues',
        orientation='h'
    )
    
    st.plotly_chart(fig_insurance, use_container_width=True)

# VISUALIZATION 6: Billing by Admission Type (Box Plot)
with col_f:
    fig_billing_admission = px.box(
        filtered_df,
        x='Admission Type',
        y='Billing Amount',
        title='Billing Amount Distribution by Admission Type',
        color='Admission Type',
        color_discrete_sequence=['#EF553B', '#00CC96', '#AB63FA']
    )
    
    st.plotly_chart(fig_billing_admission, use_container_width=True)

# ============================================
# ADDITIONAL VISUALIZATION
# ============================================
st.subheader("👨‍⚕️ Additional Analysis")

col_g, col_h = st.columns(2)

# VISUALIZATION 7: Average Billing by Medical Condition (Bar Chart)
with col_g:
    condition_billing = filtered_df.groupby('Medical Condition')['Billing Amount'].mean().reset_index()
    condition_billing = condition_billing.sort_values('Billing Amount', ascending=False)
    
    fig_condition_billing = px.bar(
        condition_billing,
        x='Medical Condition',
        y='Billing Amount',
        title='Average Billing Amount by Medical Condition',
        color='Billing Amount',
        color_continuous_scale='Reds'
    )
    fig_condition_billing.update_layout(xaxis_tickangle=-45)
    
    st.plotly_chart(fig_condition_billing, use_container_width=True)

# VISUALIZATION 8: Average LOS by Medical Condition (Bar Chart)
with col_h:
    condition_los = filtered_df.groupby('Medical Condition')['LOS'].mean().reset_index()
    condition_los = condition_los.sort_values('LOS', ascending=False)
    
    fig_condition_los = px.bar(
        condition_los,
        x='Medical Condition',
        y='LOS',
        title='Average Length of Stay by Medical Condition',
        color='LOS',
        color_continuous_scale='Greens'
    )
    fig_condition_los.update_layout(xaxis_tickangle=-45)
    
    st.plotly_chart(fig_condition_los, use_container_width=True)

# ============================================
# DETAILED STATISTICS TABLE
# ============================================
st.divider()
st.subheader("📋 Detailed Statistics by Medical Condition")

# Create summary statistics table
summary_stats = filtered_df.groupby('Medical Condition').agg({
    'Name': 'count',
    'Age': ['mean', 'min', 'max'],
    'Billing Amount': ['mean', 'min', 'max'],
    'LOS': ['mean', 'min', 'max']
}).round(2)

summary_stats.columns = ['Patient Count', 'Avg Age', 'Min Age', 'Max Age', 
                         'Avg Billing', 'Min Billing', 'Max Billing', 
                         'Avg LOS', 'Min LOS', 'Max LOS']

st.dataframe(summary_stats, use_container_width=True)

# ============================================
# STATISTICS BY HOSPITAL
# ============================================
st.subheader("🏥 Hospital Performance Statistics")

hospital_stats = filtered_df.groupby('Hospital').agg({
    'Name': 'count',
    'Billing Amount': ['mean', 'sum'],
    'LOS': 'mean',
    'Age': 'mean'
}).round(2)

hospital_stats.columns = ['Patient Count', 'Avg Billing', 'Total Billing', 'Avg LOS', 'Avg Age']
hospital_stats = hospital_stats.sort_values('Patient Count', ascending=False)

st.dataframe(hospital_stats, use_container_width=True)

# ============================================
# STATISTICS BY DOCTOR
# ============================================
st.subheader("👨‍⚕️ Doctor Performance Statistics")

doctor_stats = filtered_df.groupby('Doctor').agg({
    'Name': 'count',
    'Billing Amount': 'mean',
    'LOS': 'mean'
}).round(2)

doctor_stats.columns = ['Patient Count', 'Avg Billing', 'Avg LOS']
doctor_stats = doctor_stats.sort_values('Patient Count', ascending=False).head(10)

st.dataframe(doctor_stats, use_container_width=True)

# ============================================
# RAW DATA TABLE
# ============================================
st.divider()
st.subheader("🔎 Raw Patient Data")

# Add column selection for better readability
columns_to_display = st.multiselect(
    "Select columns to display:",
    options=filtered_df.columns.tolist(),
    default=['Name', 'Age', 'Gender', 'Medical Condition', 'Hospital', 'Doctor', 
             'Billing Amount', 'Admission Type', 'LOS', 'Test Results']
)

# Display table
st.dataframe(
    filtered_df[columns_to_display],
    use_container_width=True,
    height=400
)

# ============================================
# DOWNLOAD DATA
# ============================================
st.divider()
st.subheader("⬇️ Export Data")

# Convert to CSV
csv_data = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv_data,
    file_name=f"hospital_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# ============================================
# DATA SUMMARY
# ============================================
st.divider()
st.subheader("📊 Data Summary")

col_summary1, col_summary2, col_summary3 = st.columns(3)

with col_summary1:
    st.write("**Gender Distribution**")
    gender_counts = filtered_df['Gender'].value_counts()
    st.bar_chart(gender_counts)

with col_summary2:
    st.write("**Blood Type Distribution**")
    blood_counts = filtered_df['Blood Type'].value_counts()
    st.bar_chart(blood_counts)

with col_summary3:
    st.write("**Test Results Distribution**")
    test_counts = filtered_df['Test Results'].value_counts()
    st.bar_chart(test_counts)

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
---
**Dashboard Information:**
- 📊 This dashboard provides descriptive analytics of hospital operations
- 🔍 Use filters in the sidebar to explore specific segments
- 📈 All metrics are calculated from the selected filtered data
- 💾 Download filtered data for further analysis

**Data Columns Used:**
- Name, Age, Gender, Blood Type, Medical Condition, Date of Admission, Doctor, Hospital
- Insurance Provider, Billing Amount, Room Number, Admission Type, Discharge Date
- Medication, Test Results, LOS (calculated)

**Created for:** Informatika Terapan - Module 7: Data Analytics & Business Intelligence in Healthcare
""")

