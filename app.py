
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Progress Tracker",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

MODEL_PATH = "/content/drive/MyDrive/StudentProgressTracker/models/student_performance_model.pkl"

FEATURES_PATH = "/content/drive/MyDrive/StudentProgressTracker/models/student_performance_features.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, features


try:
    model, model_features = load_model()

except Exception as e:
    st.error("Unable to load the trained ML model.")
    st.error(str(e))
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_student_performance(
    performance_1,
    performance_2,
    performance_3,
    performance_4
):
    """
    Predict the next performance score using
    the trained student-friendly Random Forest model.

    Performance 1 = oldest
    Performance 4 = most recent
    """

    # Convert user's chronological input into
    # the order expected by the ML model.

    previous_score_1 = performance_4
    previous_score_2 = performance_3
    previous_score_3 = performance_2
    previous_score_4 = performance_1

    # Average of previous performances
    previous_average = (
        previous_score_1
        + previous_score_2
        + previous_score_3
        + previous_score_4
    ) / 4

    # Recent performance compared with oldest performance
    performance_trend = (
        previous_score_1
        - previous_score_4
    )

    input_data = pd.DataFrame([{
        "previous_score_1": previous_score_1,
        "previous_score_2": previous_score_2,
        "previous_score_3": previous_score_3,
        "previous_score_4": previous_score_4,
        "previous_average": previous_average,
        "performance_trend": performance_trend
    }])

    # Make sure features are in exactly the same
    # order used during model training.
    input_data = input_data[model_features]

    prediction = model.predict(input_data)[0]

    # Keep prediction between 0 and 100
    prediction = np.clip(prediction, 0, 100)

    return float(prediction)


def get_category(score):

    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "At Risk"


def get_trend(performance_1, performance_2,
              performance_3, performance_4):

    first_half = (
        performance_1 + performance_2
    ) / 2

    second_half = (
        performance_3 + performance_4
    ) / 2

    difference = second_half - first_half

    if difference >= 5:
        return "Improving"

    elif difference <= -5:
        return "Declining"

    else:
        return "Stable"


def get_recommendation(category, trend):

    if category == "Excellent":
        if trend == "Improving":
            return "Excellent progress. Continue the current study strategy and maintain consistency."
        elif trend == "Declining":
            return "Your current performance is strong, but the recent decline should be monitored."
        else:
            return "Performance is strong and stable. Continue your current approach."

    elif category == "Good":
        if trend == "Improving":
            return "Good performance with positive progress. Continue your current study routine."
        elif trend == "Declining":
            return "Performance is still good, but the recent decline needs attention."
        else:
            return "Performance is stable. Focus on consistent practice to move toward excellent performance."

    elif category == "Average":
        if trend == "Improving":
            return "You are improving. Increase practice time and continue focusing on weaker subjects."
        elif trend == "Declining":
            return "Performance is declining. Review difficult topics and increase regular practice."
        else:
            return "Performance is stable but can improve. Focus on consistency and targeted practice."

    else:
        if trend == "Improving":
            return "You are showing improvement, but additional practice is recommended to reach a safer performance level."
        else:
            return "Immediate attention is recommended. Review weak areas, increase practice, and seek academic support if needed."


# ============================================================
# HEADER
# ============================================================

st.title("🎓 Student Progress Tracking & Performance Prediction")

st.write(
    "Analyze historical academic performance and predict the student's "
    "next performance using a trained Random Forest machine-learning model."
)

st.divider()


# ============================================================
# STUDENT INFORMATION
# ============================================================

st.subheader("👨‍🎓 Student Information")

col1, col2 = st.columns(2)

with col1:

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student name"
    )

    student_id = st.text_input(
        "Student ID",
        placeholder="Enter student ID"
    )

    course = st.text_input(
        "Course",
        placeholder="e.g. B.Tech Computer Science"
    )


with col2:

    department = st.text_input(
        "Department",
        placeholder="e.g. Computer Engineering"
    )

    semester = st.selectbox(
        "Semester",
        [
            "Semester 1",
            "Semester 2",
            "Semester 3",
            "Semester 4",
            "Semester 5",
            "Semester 6",
            "Semester 7",
            "Semester 8"
        ]
    )

    attendance = st.number_input(
        "Overall Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    study_hours = st.number_input(
        "Average Study Hours / Day",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )


st.divider()


# ============================================================
# SUBJECT INPUT
# ============================================================

st.subheader("📚 Subject Performance")

st.info(
    "Enter the student's four most recent performance periods. "
    "These values are used by the trained Random Forest model "
    "to predict the next performance."
)

num_subjects = st.number_input(
    "Number of Subjects",
    min_value=1,
    max_value=12,
    value=3,
    step=1
)


subjects = []

for i in range(int(num_subjects)):

    st.markdown(f"### Subject {i + 1}")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        subject_name = st.text_input(
            "Subject Name",
            key=f"subject_name_{i}",
            placeholder="e.g. Mathematics"
        )

    with col2:
        performance_1 = st.number_input(
            "Period 1",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key=f"performance_1_{i}"
        )

    with col3:
        performance_2 = st.number_input(
            "Period 2",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key=f"performance_2_{i}"
        )

    with col4:
        performance_3 = st.number_input(
            "Period 3",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key=f"performance_3_{i}"
        )

    with col5:
        performance_4 = st.number_input(
            "Period 4",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key=f"performance_4_{i}"
        )

    subjects.append({
        "Subject": subject_name,
        "Period 1": performance_1,
        "Period 2": performance_2,
        "Period 3": performance_3,
        "Period 4": performance_4
    })


st.divider()


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔮 Analyze Student",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    valid_subjects = [
        subject for subject in subjects
        if subject["Subject"].strip()
    ]

    if not valid_subjects:

        st.warning(
            "Please enter at least one subject name."
        )

        st.stop()


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    results = []

    for subject in valid_subjects:

        predicted_score = predict_student_performance(
            subject["Period 1"],
            subject["Period 2"],
            subject["Period 3"],
            subject["Period 4"]
        )

        category = get_category(
            predicted_score
        )

        trend = get_trend(
            subject["Period 1"],
            subject["Period 2"],
            subject["Period 3"],
            subject["Period 4"]
        )

        recommendation = get_recommendation(
            category,
            trend
        )

        average = (
            subject["Period 1"]
            + subject["Period 2"]
            + subject["Period 3"]
            + subject["Period 4"]
        ) / 4

        results.append({
            "Subject": subject["Subject"],
            "Period 1": subject["Period 1"],
            "Period 2": subject["Period 2"],
            "Period 3": subject["Period 3"],
            "Period 4": subject["Period 4"],
            "Historical Average": average,
            "Predicted Score": predicted_score,
            "Category": category,
            "Trend": trend,
            "Recommendation": recommendation
        })


    results_df = pd.DataFrame(results)


    # ========================================================
    # RESULTS
    # ========================================================

    st.header("📊 Prediction Results")


    # --------------------------------------------------------
    # Overall prediction
    # --------------------------------------------------------

    overall_prediction = results_df[
        "Predicted Score"
    ].mean()

    overall_category = get_category(
        overall_prediction
    )


    # Overall trend
    overall_period_1 = results_df["Period 1"].mean()
    overall_period_2 = results_df["Period 2"].mean()
    overall_period_3 = results_df["Period 3"].mean()
    overall_period_4 = results_df["Period 4"].mean()

    overall_trend = get_trend(
        overall_period_1,
        overall_period_2,
        overall_period_3,
        overall_period_4
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Predicted Performance",
            f"{overall_prediction:.2f}"
        )

    with col2:
        st.metric(
            "Performance Category",
            overall_category
        )

    with col3:
        st.metric(
            "Overall Trend",
            overall_trend
        )

    with col4:
        st.metric(
            "Attendance",
            f"{attendance:.1f}%"
        )


    st.divider()


    # --------------------------------------------------------
    # Subject results
    # --------------------------------------------------------

    st.subheader("📚 Subject-wise Results")

    display_df = results_df[
        [
            "Subject",
            "Historical Average",
            "Predicted Score",
            "Category",
            "Trend"
        ]
    ].copy()

    display_df["Historical Average"] = (
        display_df["Historical Average"].round(2)
    )

    display_df["Predicted Score"] = (
        display_df["Predicted Score"].round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Performance chart
    # --------------------------------------------------------

    st.subheader("📈 Performance Trend")

    chart_data = results_df.set_index(
        "Subject"
    )[
        [
            "Period 1",
            "Period 2",
            "Period 3",
            "Period 4",
            "Predicted Score"
        ]
    ]

    st.line_chart(chart_data)


    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    st.subheader("💡 Recommendations")

    for _, row in results_df.iterrows():

        st.markdown(
            f"**{row['Subject']} — {row['Category']} / "
            f"{row['Trend']}**"
        )

        st.write(
            row["Recommendation"]
        )


    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    st.divider()

    st.subheader("🤖 Machine Learning Model")

    st.write(
        "Predictions are generated using a Random Forest "
        "Regression model trained on historical OULAD "
        "assessment data."
    )

    st.caption(
        "The deployed model uses four previous performance "
        "scores, their average, and the performance trend."
    )
