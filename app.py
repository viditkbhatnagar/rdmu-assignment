import streamlit as st
import openai
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import io
import re
from PyPDF2 import PdfReader

# -----------------------------
# Page configuration
st.set_page_config(page_title="Alarm Intelligence App", layout="wide")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -----------------------------
# Helper Functions

def extract_text_from_pdf(uploaded_pdf):
    try:
        pdf_reader = PdfReader(uploaded_pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def ask_gpt_for_analysis(case_study_text):
    """
    Sends the PDF case study text to OpenAI GPT to answer:
    1. How can the business problem be converted into an analytics problem?
    2. What key actions did Osum take to overcome the problem and why?
    3. How can a data-driven solution be achieved, based on the case study?
    """
    system_prompt = (
        "You are an expert data scientist with domain knowledge in industrial alarm management. "
        "You have been given a case study and must answer the following questions:"
        "\n1. How can the business problem be converted into an analytics problem?"
        "\n2. What key actions did Osum take to overcome the problem and why?"
        "\n3. How can a data-driven solution be achieved, based on the case study?"
    )
    user_prompt = f"Case Study:\n\n{case_study_text}\n\nPlease answer the above questions based on this case study."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        answer = response["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        return f"Error calling GPT API: {e}"

def visualize_alarm_data(df):
    """
    Expects a DataFrame with columns:
      Date Timestamp, TagName, Severity, Grouping of Alarms, Naming
    Displays several interactive visualizations.
    """
    # Ensure Date Timestamp is datetime
    if "Date Timestamp" in df.columns:
        df["Date Timestamp"] = pd.to_datetime(df["Date Timestamp"], errors="coerce")
    
    st.markdown("### Filter Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        alarm_type = st.selectbox("Select Alarm Type", options=["All"] + sorted(df["TagName"].dropna().unique().tolist()))
    with col2:
        severity = st.selectbox("Select Severity", options=["All"] + sorted(df["Severity"].dropna().unique().astype(str).tolist()))
    with col3:
        grouping = st.selectbox("Select Grouping", options=["All"] + sorted(df["Grouping of Alarms"].dropna().unique().tolist()))
    
    # Apply filters
    df_filtered = df.copy()
    if alarm_type != "All":
        df_filtered = df_filtered[df_filtered["TagName"] == alarm_type]
    if severity != "All":
        df_filtered = df_filtered[df_filtered["Severity"].astype(str) == severity]
    if grouping != "All":
        df_filtered = df_filtered[df_filtered["Grouping of Alarms"] == grouping]

    st.markdown("### Visualizations")

    # Graph 1: Alarm Count Over Time (Hourly)
    st.subheader("Alarm Count Over Time (Hourly)")
    if "Date Timestamp" in df_filtered.columns:
        df_time = df_filtered.groupby(pd.Grouper(key="Date Timestamp", freq="H")).size().reset_index(name="Count")
        fig1 = go.Figure(data=go.Scatter(x=df_time["Date Timestamp"], y=df_time["Count"],
                                         mode="lines+markers", marker_color="blue"))
        fig1.update_layout(xaxis_title="Time (Hourly)", yaxis_title="Alarm Count")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No 'Date Timestamp' column found.")

    # Graph 2: Severity Distribution (Bar Chart)
    st.subheader("Severity Distribution")
    if "Severity" in df_filtered.columns:
        severity_counts = df_filtered["Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        fig2 = go.Figure(data=go.Bar(x=severity_counts["Severity"], y=severity_counts["Count"], marker_color="red"))
        fig2.update_layout(xaxis_title="Severity", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No 'Severity' column found.")

    # Graph 3: Alarm Grouping Distribution (Pie Chart)
    st.subheader("Alarm Grouping Distribution")
    if "Grouping of Alarms" in df_filtered.columns:
        grouping_counts = df_filtered["Grouping of Alarms"].value_counts().reset_index()
        grouping_counts.columns = ["Grouping", "Count"]
        fig3 = go.Figure(data=go.Pie(labels=grouping_counts["Grouping"], values=grouping_counts["Count"]))
        fig3.update_layout(legend_title="Grouping")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No 'Grouping of Alarms' column found.")

    # Graph 4: Alarm Count by TagName (Histogram)
    st.subheader("Alarm Count by TagName")
    if "TagName" in df_filtered.columns:
        tag_counts = df_filtered["TagName"].value_counts().reset_index()
        tag_counts.columns = ["TagName", "Count"]
        fig4 = go.Figure(data=go.Bar(x=tag_counts["TagName"], y=tag_counts["Count"], marker_color="orange"))
        fig4.update_layout(xaxis_title="TagName", yaxis_title="Alarm Count")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("No 'TagName' column found.")

def parse_markov_data(df):
    """
    Expects a DataFrame where:
    - The first column is the current alarm type.
    - The remaining columns are counts for each possible next alarm type.
    Returns a row-stochastic matrix (dictionary) for Markov prediction.
    """
    try:
        df_indexed = df.set_index(df.columns[0])
        numeric_df = df_indexed.apply(pd.to_numeric, errors="coerce").fillna(0)
        row_sums = numeric_df.sum(axis=1)
        markov_matrix = {}
        for idx, row in numeric_df.iterrows():
            row_dict = {}
            total = row_sums.loc[idx]
            for col in numeric_df.columns:
                row_dict[col] = row[col] / total if total > 0 else 0
            markov_matrix[idx] = row_dict
        return markov_matrix
    except Exception as e:
        st.error(f"Error parsing Markov data: {e}")
        return None

def predict_next_alarm(current_alarm, markov_matrix):
    """
    Predicts the next alarm type from the markov_matrix given a current_alarm.
    """
    if current_alarm not in markov_matrix:
        return None
    transitions = markov_matrix[current_alarm]
    # Basic prediction: highest probability
    next_alarm = max(transitions, key=transitions.get)
    return next_alarm

def ask_gpt_markov(current_alarm, markov_matrix):
    """
    Use GPT to provide a prediction and justification for the next alarm.
    We send the current alarm and the computed matrix (as JSON) to GPT,
    and ask for the predicted next alarm type along with a detailed explanation.
    """
    prompt = (
        f"Given the following row-stochastic Markov matrix for alarm transitions:\n\n"
        f"{json.dumps(markov_matrix, indent=2)}\n\n"
        f"If the current alarm type is '{current_alarm}', "
        "predict the next alarm type and explain in detail how you arrived at that conclusion."
    )
    system_prompt = (
        "You are a data science expert specializing in Markov chain analysis for industrial alarm management. "
        "Analyze the provided Markov matrix and, based on the current alarm type, predict the next alarm type. "
        "Include a clear explanation of the reasoning."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        answer = response["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        return f"Error calling GPT for Markov prediction: {e}"

# -----------------------------
# Page Functions

def page_home():
    st.title("Alarm Intelligence App")
    st.write(
        "Welcome to the Alarm Intelligence App. Use the navigation menu on the sidebar to explore different sections:\n\n"
        "- **GPT Q&A:** Upload a PDF case study to get custom answers about the business problem, key actions, and data-driven solutions.\n"
        "- **Visualizations:** Upload alarm data (Excel file) and explore multiple interactive visualizations.\n"
        "- **Markov Chain:** Upload a transition matrix (Excel file) to compute a Markov model, predict the next alarm type, and receive a detailed explanation along with additional visuals.\n"
        "- **Conclusion & Impact:** Read about the potential impact and future directions of advanced alarm management."
    )
    try:
        st.video("industrial_background.mp4", start_time=0, use_container_width=True)
    except Exception:
        st.image("https://images.unsplash.com/photo-1593642634367-d91a135587b5?auto=format&fit=crop&w=1350&q=80", use_container_width=True)

def page_gpt_qna():
    st.title("GPT-Powered Q&A")
    st.write(
        "Upload a PDF file containing your case study. Based on this case study, GPT will answer the following questions:\n\n"
        "1. How can the business problem be converted into an analytics problem?\n"
        "2. What key actions did Osum take to overcome the problem and why?\n"
        "3. How can a data-driven solution be achieved for this case?"
    )
    
    # Initialize session state variables if they don't exist
    if "gpt_response" not in st.session_state:
        st.session_state.gpt_response = None
    if "case_study_text" not in st.session_state:
        st.session_state.case_study_text = None

    openai_key = st.text_input("Enter your OpenAI API Key", type="password", key="gpt_api_key")
    uploaded_pdf = st.file_uploader("Upload PDF Case Study", type=["pdf"], key="pdf_upload")
    
    if st.button("Ask GPT", key="ask_gpt"):
        if not openai_key:
            st.error("Please enter your OpenAI API key.")
        elif not uploaded_pdf:
            st.error("Please upload a PDF file with your case study.")
        else:
            openai.api_key = openai_key
            case_study_text = extract_text_from_pdf(uploaded_pdf)
            if case_study_text:
                st.session_state.case_study_text = case_study_text
                st.markdown("### Case Study Text Preview")
                st.write(case_study_text[:500] + "...")
                answer = ask_gpt_for_analysis(case_study_text)
                st.session_state.gpt_response = answer
                st.markdown("### GPT Response")
                st.write(answer)
                # Static findings container below the GPT response
                st.markdown("### Our Findings")
                findings_container = """
                <div style="height:600px; overflow-y:auto; border:1px solid #ddd; padding:10px;">
                <strong>a) Convert the Business Problem to an Analytics Problem.</strong><br>
                <em>Ans):</em><br>
                The problem: Osum Oil Sands Corp's SAGD plant faced an issue of alarm flooding and chattering on December 11, 2020 due to repetitive alarms triggered by interconnected systems. 772 alarms (95% repetitive), overwhelmed the operators, leading to delayed responses and near outages and risks of costly downtime.<br><br>
                How is this a Business problem: The failure of alarm management system can directly impact Osum's operational efficiency, safety, and profitability. How?:<br>
                - Operational inefficiency: Operators waste time on repetitive alarms instead of critical tasks.<br>
                - Safety risks: Missed critical alarms could lead to equipment failure or accidents and possible downtime of the plant.<br>
                - Financial loss: Unplanned downtime costs millions in lost production.<br>
                - Reputation: Frequent outages could harm Osum’s standing in the oil industry.<br><br>
                <strong>Conversion to an Analytics Problem:</strong><br>
                - Business problem: Reduce chattering alarms and prioritize critical alerts.<br>
                  Analytics problem: Classification: Predict whether an alarm is a "chattering" event (binary classification).<br>
                - Business problem: Predict which alarm will occur next to enable proactive intervention.<br>
                  Analytics problem: Sequence Prediction: Model alarm sequences to forecast the next alarm type.<br>
                - Business problem: Minimize alarm floods and operator workload.<br>
                  Analytics problem: Clustering & Prioritization: Group alarms by severity and root cause.<br>
                - Business problem: Reduce false alarms (nuisance alarms).<br>
                  Analytics problem: Anomaly Detection: Filter out redundant/stale alarms.<br><br>
                Thus, by translating Osum’s alarm management crisis into classification, sequence prediction, and anomaly detection tasks, a high-stakes business problem can be converted into a structured analytics workflow. This approach enables proactive alarm handling, reduces downtime risks, and aligns with Osum’s operational goals.<br><br>
                
                <strong>b) What were the key actions Osum took to overcome and why?</strong><br>
                <em>Ans):</em><br>
                Key actions taken by OSUM to overcome these challenges:<br>
                1. It partnered with Drishya AI Labs, an AI startup specializing in energy sector solutions, to develop a machine learning-based alarm management system. OSUM management was aware that this problem can be solved efficiently and effectively by implementing AI. However, they lacked in-house expertise in AI which was required to address the complex alarm floods and chattering issue. Bringing in Drishya AI Labs with domain knowledge in energy and AI capabilities offered them a tailored solution.<br>
                2. Prioritized critical systems and shared the information with Drishya AI Labs for effective solution. OSUM suggested them to focus on evaporator system as it was one of the critical systems in the plant. This approach could subsequently be extended to other systems in the plant. Addressing this high-risk area first allowed for quick wins and proof-of-concept validation.<br>
                3. They leveraged industry standards. They adopted ANSI/ISA 18.2 standards to define chattering (≥3 alarms/minute) and alarm floods (>10 alarms/10 minutes). These standardized definitions ensured alignment with global best practices and simplified model training. It provided clear thresholds for classifying nuisance alarms, reducing subjectivity.<br>
                4. They provided historical data. They shared 3 years of historical alarm data which was essential for training ML models to identify patterns (chattering trends, alarm sequences). Historical context also enabled predictive analytics (e.g., forecasting next alarms).<br>
                5. They clearly defined their analytical objectives to structure the problem into two analytics tasks:<br>
                   - Classification: Predict chattering alarms (CHB = 1).<br>
                   - Sequence Prediction: Forecast the next alarm type based on historical sequences.<br>
                This classification reduced operator distraction by filtering non-critical alarms. Sequence prediction enabled proactive interventions to prevent downtime.<br><br>
                
                <strong>c) How can a data-driven solution to the business problem be achieved?</strong><br>
                <em>Ans):</em><br>
                Achieving a Data-Driven Solution to Osum’s Business Problem involves systematically leveraging historical data, machine learning (ML), and analytics to reduce nuisance alarms, prioritize critical alerts, and prevent downtime. Here’s a structured approach:<br>
                1. Define the Business Objective<br>
                   - Goal: Reduce alarm floods/chattering, improve operator efficiency, and minimize downtime risks.<br>
                   - Success Metrics: 30–50% reduction in chattering alarms (≥3 alarms/minute), alarm floods reduced to <1% of total operational time, zero unplanned downtime events due to missed alarms.<br>
                2. Data Collection & Preparation<br>
                   - Gather Historical Data: Alarm logs (3 years) with variables like TagName, Severity, Alarm Tag Type, CHB, ATD, and time-based features.<br>
                   - Preprocess & Engineer Features: Clean data, remove duplicates, handle missing values, and label chattering alarms using ANSI/ISA 18.2 thresholds.<br>
                3. Model Development<br>
                   - Task 1: Classify Chattering Alarms using supervised models (e.g., Random Forest, XGBoost).<br>
                   - Task 2: Predict Next Alarm Type using sequence prediction models (e.g., LSTM networks, Markov Chains).<br>
                   - Task 3: Prioritize Critical Alarms via clustering and prioritization techniques (e.g., K-means clustering).<br>
                4. Validation & Testing<br>
                   - Use cross-validation, performance metrics (precision, recall, F1-score, accuracy), and A/B testing to ensure robustness.<br>
                5. Deployment & Integration<br>
                   - Integrate models with the DCS to provide real-time alerts, priority dashboards, and predictive warnings.<br>
                6. Monitoring & Iteration<br>
                   - Implement feedback loops, monitor for drift, and continuously retrain models with updated data.<br><br>
                By combining domain expertise with advanced analytics, this data-driven solution transforms raw alarm data into actionable insights, aligning with Osum’s operational and financial goals.
                </div>
                """
                st.markdown(findings_container, unsafe_allow_html=True)
    
    elif st.session_state.gpt_response:
        st.markdown("### GPT Response")
        st.write(st.session_state.gpt_response)
        st.markdown("### Our Findings")
        findings_container = """
        <div style="height:600px; overflow-y:auto; border:1px solid #ddd; padding:10px;">
        <strong>a) Convert the Business Problem to an Analytics Problem.</strong><br>
        <em>Ans):</em><br>
        The problem: Osum Oil Sands Corp's SAGD plant faced an issue of alarm flooding and chattering on December 11, 2020 due to repetitive alarms triggered by interconnected systems. 772 alarms (95% repetitive), overwhelmed the operators, leading to delayed responses and near outages and risks of costly downtime.<br><br>
        How is this a Business problem: The failure of alarm management system can directly impact Osum's operational efficiency, safety, and profitability. How?:<br>
        - Operational inefficiency: Operators waste time on repetitive alarms instead of critical tasks.<br>
        - Safety risks: Missed critical alarms could lead to equipment failure or accidents and possible downtime of the plant.<br>
        - Financial loss: Unplanned downtime costs millions in lost production.<br>
        - Reputation: Frequent outages could harm Osum’s standing in the oil industry.<br><br>
        <strong>Conversion to an Analytics Problem:</strong><br>
        - Business problem: Reduce chattering alarms and prioritize critical alerts.<br>
          Analytics problem: Classification: Predict whether an alarm is a "chattering" event (binary classification).<br>
        - Business problem: Predict which alarm will occur next to enable proactive intervention.<br>
          Analytics problem: Sequence Prediction: Model alarm sequences to forecast the next alarm type.<br>
        - Business problem: Minimize alarm floods and operator workload.<br>
          Analytics problem: Clustering & Prioritization: Group alarms by severity and root cause.<br>
        - Business problem: Reduce false alarms (nuisance alarms).<br>
          Analytics problem: Anomaly Detection: Filter out redundant/stale alarms.<br><br>
        Thus, by translating Osum’s alarm management crisis into classification, sequence prediction, and anomaly detection tasks, a high-stakes business problem can be converted into a structured analytics workflow. This approach enables proactive alarm handling, reduces downtime risks, and aligns with Osum’s operational goals.<br><br>
        
        <strong>b) What were the key actions Osum took to overcome and why?</strong><br>
        <em>Ans):</em><br>
        Key actions taken by OSUM to overcome these challenges:<br>
        1. It partnered with Drishya AI Labs, an AI startup specializing in energy sector solutions, to develop a machine learning-based alarm management system. OSUM management was aware that this problem can be solved efficiently and effectively by implementing AI. However, they lacked in-house expertise in AI which was required to address the complex alarm floods and chattering issue. Bringing in Drishya AI Labs with domain knowledge in energy and AI capabilities offered them a tailored solution.<br>
        2. Prioritized critical systems and shared the information with Drishya AI Labs for effective solution. OSUM suggested them to focus on evaporator system as it was one of the critical systems in the plant. This approach could subsequently be extended to other systems in the plant. Addressing this high-risk area first allowed for quick wins and proof-of-concept validation.<br>
        3. They leveraged industry standards. They adopted ANSI/ISA 18.2 standards to define chattering (≥3 alarms/minute) and alarm floods (>10 alarms/10 minutes). These standardized definitions ensured alignment with global best practices and simplified model training. It provided clear thresholds for classifying nuisance alarms, reducing subjectivity.<br>
        4. They provided historical data. They shared 3 years of historical alarm data which was essential for training ML models to identify patterns (chattering trends, alarm sequences). Historical context also enabled predictive analytics (e.g., forecasting next alarms).<br>
        5. They clearly defined their analytical objectives to structure the problem into two analytics tasks:<br>
           - Classification: Predict chattering alarms (CHB = 1).<br>
           - Sequence Prediction: Forecast the next alarm type based on historical sequences.<br>
        This classification reduced operator distraction by filtering non-critical alarms. Sequence prediction enabled proactive interventions to prevent downtime.<br><br>
        
        <strong>c) How can a data-driven solution to the business problem be achieved?</strong><br>
        <em>Ans):</em><br>
        Achieving a Data-Driven Solution to Osum’s Business Problem involves systematically leveraging historical data, machine learning (ML), and analytics to reduce nuisance alarms, prioritize critical alerts, and prevent downtime. Here’s a structured approach:<br>
        1. Define the Business Objective<br>
           - Goal: Reduce alarm floods/chattering, improve operator efficiency, and minimize downtime risks.<br>
           - Success Metrics: 30–50% reduction in chattering alarms (≥3 alarms/minute), alarm floods reduced to <1% of total operational time, zero unplanned downtime events due to missed alarms.<br>
        2. Data Collection & Preparation<br>
           - Gather Historical Data: Alarm logs (3 years) with variables like TagName, Severity, Alarm Tag Type, CHB, ATD, and time-based features.<br>
           - Preprocess & Engineer Features: Clean data, remove duplicates, handle missing values, and label chattering alarms using ANSI/ISA 18.2 thresholds.<br>
        3. Model Development<br>
           - Task 1: Classify Chattering Alarms using supervised models (e.g., Random Forest, XGBoost).<br>
           - Task 2: Predict Next Alarm Type using sequence prediction models (e.g., LSTM networks, Markov Chains).<br>
           - Task 3: Prioritize Critical Alarms via clustering and prioritization techniques (e.g., K-means clustering).<br>
        4. Validation & Testing<br>
           - Use cross-validation, performance metrics (precision, recall, F1-score, accuracy), and A/B testing to ensure robustness.<br>
        5. Deployment & Integration<br>
           - Integrate models with the DCS to provide real-time alerts, priority dashboards, and predictive warnings.<br>
        6. Monitoring & Iteration<br>
           - Implement feedback loops, monitor for drift, and continuously retrain models with updated data.<br><br>
        By combining domain expertise with advanced analytics, this data-driven solution transforms raw alarm data into actionable insights, aligning with Osum’s operational and financial goals.
        </div>
        """
        st.markdown(findings_container, unsafe_allow_html=True)

def page_visualizations():
    st.title("Visualizations")
    st.write(
        "Upload an Excel file with the following columns:\n\n"
        "**Date Timestamp, TagName, Severity, Grouping of Alarms, Naming**\n\n"
        "Use the in-page filters below to explore the data interactively."
    )
    uploaded_file = st.file_uploader("Upload Excel File for Visualization", type=["xlsx"], key="viz_upload")
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.viz_df = df
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
    
    if "viz_df" in st.session_state and st.session_state.viz_df is not None:
        df = st.session_state.viz_df
        st.markdown("### Data Preview")
        st.dataframe(df.head())
        visualize_alarm_data(df)

def page_markov():
    st.title("Markov Chain Alarm Prediction")
    st.write(
        "Upload an Excel file that contains a matrix of counts. The file should be formatted as follows:\n\n"
        "• The first column is the current alarm type.\n"
        "• The remaining columns represent counts of occurrences for each next alarm type.\n\n"
        "For example, a table with rows for 'Others', 'Level', 'Flow', 'Pressure', and 'Temperature'."
    )
    uploaded_file = st.file_uploader("Upload Excel File for Markov Matrix", type=["xlsx"], key="markov_upload")
    
    # If the Excel file has been uploaded previously, derive alarm options from its first column.
    if "markov_df" in st.session_state and st.session_state.markov_df is not None:
        alarm_options = list(st.session_state.markov_df.iloc[:, 0].unique())
    else:
        alarm_options = ["Others", "Level", "Flow", "Pressure", "Temperature"]
    
    current_alarm = st.selectbox("Enter Current Alarm Type", options=alarm_options, key="current_alarm")
    
    openai_key = st.text_input("Enter your OpenAI API Key for Markov Explanation (optional)", type="password", key="markov_api_key")
    
    if st.button("Compute and Predict", key="compute_markov"):
        if not uploaded_file and "markov_df" not in st.session_state:
            st.error("Please upload an Excel file first.")
            return
        try:
            # Read and store Markov raw data
            if uploaded_file is not None:
                df = pd.read_excel(uploaded_file)
                st.session_state.markov_df = df
            else:
                df = st.session_state.markov_df
            st.markdown("### Raw Markov Data Preview")
            st.dataframe(df.head())
            markov_matrix = parse_markov_data(df)
            if markov_matrix is None:
                return
            st.session_state.markov_matrix = markov_matrix
            st.markdown("### Row-Stochastic Markov Matrix")
            st.json(markov_matrix)
            
            local_prediction = predict_next_alarm(current_alarm, markov_matrix)
            st.session_state.local_prediction = local_prediction
            st.subheader("Local Prediction")
            st.write(f"Predicted Next Alarm (local calculation): **{local_prediction}**")
            
            if openai_key:
                openai.api_key = openai_key
                gpt_response = ask_gpt_markov(current_alarm, markov_matrix)
                st.session_state.markov_gpt_response = gpt_response
                st.subheader("GPT Prediction & Justification")
                st.write(gpt_response)
            else:
                st.info("To get a detailed GPT explanation, please enter your OpenAI API key above.")
            
            # -----------------------------
            # Additional Visualizations Section
            st.markdown("### Additional Visuals for Local Prediction")
            if current_alarm in markov_matrix:
                row_data = markov_matrix[current_alarm]
                alarm_types = list(row_data.keys())
                probabilities = list(row_data.values())
                
                # Bar Chart
                fig_bar = go.Figure(data=go.Bar(
                    x=alarm_types, y=probabilities,
                    marker_color='mediumseagreen'
                ))
                # Attempt to extract GPT predicted alarm from response using a simple heuristic.
                gpt_pred = "N/A"
                if st.session_state.get("markov_gpt_response"):
                    match = re.search(r"predicted next alarm\s*[:\-]?\s*(\w+)", st.session_state.markov_gpt_response, re.IGNORECASE)
                    if match:
                        gpt_pred = match.group(1)
                fig_bar.update_layout(
                    title=f"Local Prediction: Probability Distribution<br>(Local Predicted: {local_prediction}; GPT Predicted: {gpt_pred})",
                    xaxis_title="Next Alarm Type", yaxis_title="Probability", bargap=0.2
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pie Chart
                fig_pie = go.Figure(data=go.Pie(
                    labels=alarm_types, values=probabilities,
                    marker_colors=['lightskyblue', 'lightcoral', 'lightgreen', 'plum', 'khaki']
                ))
                fig_pie.update_layout(title="Local Prediction: Pie Chart of Probabilities")
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Radar Chart (Spider Chart)
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=probabilities, theta=alarm_types, fill='toself',
                    name="Probability", marker_color='dodgerblue'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, max(probabilities)*1.2])),
                    title="Local Prediction: Radar Chart"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.warning("Current alarm type not found in the Markov matrix.")
        except Exception as e:
            st.error(f"Error processing the Markov data: {e}")
    else:
        # If prediction results already exist in session state, display them.
        if "markov_matrix" in st.session_state:
            st.markdown("### Row-Stochastic Markov Matrix")
            st.json(st.session_state.markov_matrix)
        if "local_prediction" in st.session_state:
            st.subheader("Local Prediction")
            st.write(f"Predicted Next Alarm (local calculation): **{st.session_state.local_prediction}**")
        if "markov_gpt_response" in st.session_state:
            st.subheader("GPT Prediction & Justification")
            st.write(st.session_state.markov_gpt_response)
        # Additionally, display visuals if Markov matrix is available.
        if "markov_matrix" in st.session_state and st.session_state.markov_matrix.get(current_alarm):
            row_data = st.session_state.markov_matrix[current_alarm]
            alarm_types = list(row_data.keys())
            probabilities = list(row_data.values())
            
            st.markdown("### Additional Visuals for Local Prediction")
            fig_bar = go.Figure(data=go.Bar(
                x=alarm_types, y=probabilities,
                marker_color='mediumseagreen'
            ))
            fig_bar.update_layout(
                title="Local Prediction: Probability Distribution",
                xaxis_title="Next Alarm Type", yaxis_title="Probability", bargap=0.2
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            fig_pie = go.Figure(data=go.Pie(
                labels=alarm_types, values=probabilities,
                marker_colors=['lightskyblue', 'lightcoral', 'lightgreen', 'plum', 'khaki']
            ))
            fig_pie.update_layout(title="Local Prediction: Pie Chart of Probabilities")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=probabilities, theta=alarm_types, fill='toself',
                name="Probability", marker_color='dodgerblue'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(probabilities)*1.2])),
                title="Local Prediction: Radar Chart"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

def page_conclusion():
    st.title("Conclusion & Impact")
    st.markdown(
        """
        **Summary and Future Outlook:**
        
        Our exploration demonstrates that by leveraging advanced data analytics and machine learning techniques, 
        industrial alarm management can be transformed from a reactive process into a proactive, data-driven system.
        
        - **Business to Analytics Conversion:**  
          The transformation of complex operational challenges into measurable analytics problems enables organizations 
          to identify inefficiencies and prioritize areas for improvement. The case study analysis shows how alarm classification, 
          frequency analysis, and predictive modeling converge to provide actionable insights.
        
        - **Key Actions by Osum:**  
          By integrating data from Distributed Control Systems and performing rigorous feature engineering, 
          Osum overcame the limitations of traditional alarm systems. Their collaboration with AI experts streamlined operations 
          and reduced downtime, proving that targeted interventions can yield significant operational benefits.
        
        - **Data-Driven Solutions:**  
          Interactive visualizations and predictive models, including advanced Markov chain analysis, underscore the power of real-time data in decision making. 
          The Markov model not only forecasts the next alarm type but, when combined with expert GPT-driven insights, provides detailed justifications that help stakeholders understand the underlying patterns.
        
        **Looking Ahead:**  
        Future enhancements could include:
        - Integration with live data feeds for real-time monitoring.
        - Expansion of predictive models to incorporate multi-step transitions.
        - Development of an even more comprehensive user interface for deeper drill-down analyses.
        - Continuous improvement through model updates and user feedback.
        
        In conclusion, this integrated approach transforms alarm management into a strategic asset, paving the way for enhanced operational excellence and informed decision-making.
        """
    )

# -----------------------------
# Main Navigation

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Go to", ["Home", "GPT Q&A", "Visualizations", "Markov Chain", "Conclusion & Impact"])
    
    pages = {
        "Home": page_home,
        "GPT Q&A": page_gpt_qna,
        "Visualizations": page_visualizations,
        "Markov Chain": page_markov,
        "Conclusion & Impact": page_conclusion
    }
    pages[choice]()

if __name__ == "__main__":
    main()
