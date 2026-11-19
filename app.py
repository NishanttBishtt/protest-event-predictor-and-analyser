
import sys
import json
import streamlit as st

# Allow importing modules from src/
sys.path.append("src")

from src.run_event_pipeline import process_tweet


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Protest Intelligence System",
    page_icon="🌍",
    layout="wide"
)


# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .warning-box {
        background-color: #422006;
        border-left: 5px solid #F59E0B;
        padding: 12px;
        border-radius: 5px;
        color: #FDE68A;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🌍 Social Media Civil Unrest Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Protest Detection • Event Information Extraction • '
    'Country Resolution • Historical Forecasting</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- INPUT ----------------

st.markdown("### Enter Social Media Post")

tweet = st.text_area(
    "Paste a tweet for analysis",
    height=140,
    placeholder="Enter a tweet describing a protest, demonstration, rally, etc."
)

analyse = st.button(
    "Analyse Tweet",
    type="primary",
    use_container_width=True
)


# ---------------- PIPELINE EXECUTION ----------------

if analyse:

    if not tweet.strip():
        st.warning("Please enter a tweet first.")

    else:
        with st.spinner("Running the intelligence pipeline..."):

            try:
                result = process_tweet(tweet.strip())

                st.session_state["analysis_result"] = result

            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                st.stop()


# ---------------- DISPLAY RESULTS ----------------

if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]

    st.divider()
    st.header("Analysis Results")

    # Event Detection
    st.markdown("### 1. Event Detection")

    col1, col2 = st.columns(2)

    with col1:
        if result["is_event"]:
            st.success("Event Detected")
        else:
            st.error("No Event Detected")

    with col2:
        st.metric(
            "Event Probability",
            f'{result["event_probability"] * 100:.2f}%'
        )

    if result.get("review_required"):
        st.warning("This prediction may require manual review.")

    if not result["is_event"]:
        st.info("The remaining pipeline stages were skipped because no event was detected.")

    else:

        # Information Extraction
        info = result["event_information"]

        st.markdown("### 2. Event Information Extraction")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Event Type**")
            st.write(info.get("event_type") or "Not identified")

            st.write("**People**")
            st.write(", ".join(info.get("people", [])) or "Not identified")

            st.write("**Organisations**")
            st.write(", ".join(info.get("organisations", [])) or "Not identified")

            st.write("**Groups**")
            st.write(", ".join(info.get("groups", [])) or "Not identified")

        with col2:
            st.write("**Locations**")
            st.write(", ".join(info.get("locations", [])) or "Not identified")

            st.write("**Venues**")
            st.write(", ".join(info.get("venues", [])) or "Not identified")

            st.write("**Dates**")
            st.write(", ".join(info.get("dates", [])) or "Not identified")

            st.write("**Times**")
            st.write(", ".join(info.get("times", [])) or "Not identified")

        st.write("**Protest Cause**")
        st.write(info.get("protest_cause") or "Not identified")

        with st.expander("View complete extracted information"):
            st.json(info)

        # Country Resolution
        st.markdown("### 3. Country Resolution")

        country = result.get("country_resolution", {})

        if country.get("status") == "resolved":

            st.success(f'Country identified: {country.get("country_code")}')

            if country.get("evidence"):
                st.write("**Location matching evidence**")
                st.json(country["evidence"])

        elif country.get("status") == "ambiguous":
            st.warning("Multiple possible countries identified.")
            st.json(country)

        else:
            st.warning("Country could not be resolved.")
            st.json(country)

        # Forecasting
        st.markdown("### 4. Protest Forecasting")

        forecast = result.get("forecast")

        if forecast:

            if forecast.get("is_synthetic"):

                st.warning(
                    "Synthetic demonstration only: "
                    "Bangladesh-derived data is being used as a proxy for India. "
                    "This is NOT a real India forecast."
                )

            if forecast.get("error"):

                st.error(f'Forecast unavailable: {forecast["error"]}')

            else:

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Historical Model Probability",
                        f'{forecast["historical_model_probability"] * 100:.2f}%'
                    )

                with col2:
                    prediction = forecast["predicted_protest"]

                    st.metric(
                        "Prediction",
                        "Protest Expected" if prediction == 1 else "No Protest Expected"
                    )

                with col3:
                    st.metric(
                        "Model Threshold",
                        f'{forecast["threshold"] * 100:.0f}%'
                    )

                st.write(f'**Country:** {forecast.get("display_name", forecast.get("country"))}')
                st.write(f'**Data period:** {forecast.get("data_period")}')
                st.write(f'**Latest data date:** {forecast.get("data_date")}')

                st.write("**Historical protest counts**")

                c1, c2, c3 = st.columns(3)

                c1.metric("Past 7 Days", forecast["past_7d_events"])
                c2.metric("Past 14 Days", forecast["past_14d_events"])
                c3.metric("Past 28 Days", forecast["past_28d_events"])

        else:
            st.info("Forecasting skipped because the country could not be resolved.")

        # Complete JSON
        st.divider()

        with st.expander("View Complete Pipeline JSON"):
            st.json(result)

        json_data = json.dumps(result, indent=4, ensure_ascii=False)

        st.download_button(
            label="Download Analysis JSON",
            data=json_data,
            file_name="protest_analysis.json",
            mime="application/json"
        )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Research Prototype | Social-Media-Based Civil Unrest Intelligence System"
)
