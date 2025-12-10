import uvicorn
import os
import pandas as pd
import json
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile

# === CORE IMPORTS ===
from google import genai 
from google.genai import types
from dotenv import load_dotenv 
from pydantic import BaseModel, Field
from typing import Optional, List # Added List import

# =========================
# Pydantic Schema for LLM Response (CRITICAL FIX)
# This model forces the LLM to structure its output perfectly.
# =========================

class IndividualImpactData(BaseModel):
    name: str = Field(..., description="The Name of the employee.")
    overall_impact_rating: int = Field(..., description="The score from 'Rate of the intervention’s overall impact to the efficiency of the personnel' (1-5).")
    retake_decision: str = Field(..., description="The decision based on the Retake Rule (YES or NO).")

class GeminiAnalysisResponse(BaseModel):
    # Main insights, always present
    key_insights_and_prioritization: str = Field(..., description="A detailed markdown summary covering sections 1 (Key Insights & Gaps) and 2 (Recommendations for Training Prioritization).")
    
    # NEW: Structured list for individual impact assessment
    individual_impact_list: Optional[List[IndividualImpactData]] = Field(None, description="A list of JSON objects containing the individual impact metrics (Rating and Retake Decision). Null if no impact data provided.")

    # Group Impact Assessment results, remains unstructured for complex text summary
    impact_assessment_details: Optional[str] = Field(None, description="The full markdown analysis of Group Retake Summary and Future Interventions.")

# =========================
# Environment & Client Setup
# =========================
load_dotenv(dotenv_path='config.env')
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("FATAL WARNING: GEMINI_API_KEY not found. Please check config.env.")
    client = None
else:
    # Client Initialization
    client = genai.Client(api_key=API_KEY)


# =========================
# Training Configuration
# =========================
TRAINING_MAP = {
    "CK": "Orientation seminar on content-based knowledge (CK)",
    "FK": "Training on functional know-how (administration services - FK)",
    "SK": "Conceptual training on specialized topics (academic programs - SK)",
    "OS": "Practical/work-based skill trainings (organizational effectiveness - OS)",
    "FS": "Practical/work-based skill trainings (organizational effectiveness - FS)",
    "SMS": "Practical/work-based skill training (effective personal management - SMS)",
    "AW": "Trainings related to further development of attitude and work effectiveness (AW)",
    "ACW": "Trainings related to further development of attitude and work relationship (ACW)",
    "ACS": "Trainings related to further development of attitude and customer service (ACS)"
}
COMPONENTS = list(TRAINING_MAP.keys())

# --- NEW IMPACT ASSESSMENT COLUMNS ---
IMPACT_COLS = [
    "Training Plan",
    "Intervention Type",
    "Was the intervention beneficial to the personnel’s scope of work?",
    "Did the personnel incorporate the things they learned in the intervention into their work?",
    "Did you notice a significant change at your personnel’s perception, attitude or behavior as a result of the intervention?",
    "Rate of the intervention’s overall impact to the efficiency of the personnel"
]

# =========================
# Excel loader & analysis functions (Internal, remains unchanged)
# =========================
def load_excel(filepath):
    with pd.ExcelFile(filepath) as xls:
        all_sheets = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            for col in df.columns:
                if any(col.startswith(comp) for comp in COMPONENTS):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.fillna(0, inplace=True)
            all_sheets.append(df)
    return pd.concat(all_sheets, ignore_index=True)

def load_multiple_excels(filepaths):
    all_files = []
    for file in filepaths:
        try:
            df = load_excel(file)
            all_files.append(df)
        except Exception as e:
            print(f"⚠️ Error loading file '{file}': {e}")
    if all_files:
        return pd.concat(all_files, ignore_index=True)
    return pd.DataFrame()

def calculate_individual_means(df):
    for comp in COMPONENTS:
        comp_cols = [col for col in df.columns if col.startswith(comp)]
        df[comp + "_Avg"] = df[comp_cols].mean(axis=1)
    return df

def get_overall_rating(row):
    avg_cols = [comp + "_Avg" for comp in COMPONENTS]
    return row[avg_cols].mean()

def overall_top_3(df):
    avg_cols = [comp + "_Avg" for comp in COMPONENTS]
    mean_scores = df[avg_cols].mean()
    sorted_means = mean_scores.nsmallest(3)
    return [
        {
            "Rank": rank,
            "Focus_Area_Component": comp.replace("_Avg", ""),
            "Training_Recommendation": TRAINING_MAP[comp.replace("_Avg", "")],
            "Mean_Score": round(score, 2)
        }
        for rank, (comp, score) in enumerate(sorted_means.items(), start=1)
    ]

def individual_training_plans(df):
    plans = []
    for _, row in df.iterrows():
        avg_cols = [comp + "_Avg" for comp in COMPONENTS]
        lowest_col = row[avg_cols].idxmin()
        comp_name = lowest_col.replace("_Avg", "")
        name = row.get("Name", "N/A")
        position = row.get("Position", "N/A")
        office_college = row.get("Office/College", "N/A")
        plans.append({
            "Name": name,
            "Position": position,
            "Office_College": office_college,
            "Overall_Rating": round(get_overall_rating(row), 2),
            "Lowest_Component": comp_name,
            "Training_Recommendation": TRAINING_MAP[comp_name]
        })
    return plans

def group_mean_plans(df, group_col):
    groups = df.groupby(group_col)
    result = []
    avg_cols = [comp + "_Avg" for comp in COMPONENTS]
    for name, group in groups:
        group_means = group[avg_cols].mean()
        top3 = group_means.nsmallest(3)
        if top3.empty:
            continue
        result.append({
            "Group_Name": name,
            "Overall_Group_Rating": round(group_means.mean(), 2),
            "Primary_Focus_1": f"{TRAINING_MAP[top3.index[0].replace('_Avg','')]} ({round(top3.iloc[0],2)})",
            "Secondary_Focus_2": f"{TRAINING_MAP[top3.index[1].replace('_Avg','')]} ({round(top3.iloc[1],2)})",
            "Tertiary_Focus_3": f"{TRAINING_MAP[top3.index[2].replace('_Avg','')]} ({round(top3.iloc[2],2)})",
        })
    return result

def percentage_breakdowns(df, group_category):
    result = []
    groups = df.groupby(group_category)
    for name, group in groups:
        lowest_comps = group[[comp + "_Avg" for comp in COMPONENTS]].idxmin(axis=1)
        counts = lowest_comps.apply(lambda x: x.replace("_Avg","")).value_counts()
        total = len(group)
        for comp, count in counts.items():
            result.append({
                "Group_Category": group_category,
                "Group_Name": name,
                "Training_Plan": TRAINING_MAP[comp],
                "Count": int(count),
                # This is the line that generates the float percentage without "%"
                "Percentage": round(count/total*100,2) 
            })
    return result

# =========================
# FastAPI setup
# =========================
app = FastAPI(title="CATNA Analysis Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Helper to call Gemini with retries
# =========================
def call_gemini_with_retries_structured(user_message, structured_schema, retries=5, delay=5):
    if client is None:
        return {"error": "Gemini client is not initialized due to missing API Key."}
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=structured_schema,
        # NOTE: We keep the system instruction empty here as the full prompt is in the user_message
    )

    last_error = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[user_message],
                config=config,
            )
            return json.loads(response.text.strip())
        except Exception as e:
            last_error = e
            time.sleep(delay)
            delay *= 2 
            
    print(f"Gemini API unavailable after {retries} retries. Last error: {last_error}")
    return {"error": f"Gemini AI is currently unavailable after retries. (Last Error: {last_error})"}

# =========================
# API Endpoint
# =========================
@app.post("/analyze/")
async def analyze_excel(file: UploadFile = File(...)):
    if not API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Server configuration error: Gemini API Key is missing from config.env."
        )

    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files are accepted.")

    tmp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Run PURE DATA analysis
        df = load_multiple_excels([tmp_path])
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid data found in Excel file.")
            
        # Ensure key columns exist
        for col in ["Office/College", "Supervisor Name", "Position", "Campus", "Years in Current Position", "Name"]:
            if col not in df.columns:
                df[col] = "N/A"
                
        df = calculate_individual_means(df)

        # --- CHECK FOR IMPACT DATA ---
        has_impact_data = all(col in df.columns for col in IMPACT_COLS)
        
        # Determine mandatory columns for LLM input
        mandatory_cols = ["Name", "Position", "Office/College", "Campus"]
        if has_impact_data:
             mandatory_cols.extend(IMPACT_COLS)
        
        # Combine all necessary data
        all_cols = list(set(mandatory_cols + [c + "_Avg" for c in COMPONENTS] + 
                             ["Overall_Rating", "Lowest_Component", "Training_Recommendation"]))

        # Filter the DataFrame for only the columns we need to send to the LLM
        df_llm = df[[col for col in all_cols if col in df.columns]].copy()
        
        # Serialize data for LLM
        data_to_send = df_llm.to_json(orient='records', index=False)

        # --- LLM INSIGHT GENERATION PROMPT CONSTRUCTION (Unified) ---
        
        prompt_impact_instructions = "The user has NOT provided Impact Assessment data, so the impact_assessment_details field MUST be set to Null and the individual_impact_list MUST be set to Null."
        
        if has_impact_data:
            # --- MODIFIED INSTRUCTIONS FOR INDIVIDUAL LIST AND SUMMARY ---
            prompt_impact_instructions = f"""
The user has provided Impact Assessment data. Please perform the detailed analysis.

1. **Individual Impact List (For individual_impact_list field):** Generate a JSON list of objects for all employees in the Raw Employee Data who have complete impact assessment columns. For each employee, determine the Retake Decision (YES/NO) based on the Retake Rule.
2. **Group Assessment Summary (For impact_assessment_details field):** Provide a markdown summary including:
    a. Group Retake Summary: List all employees recommended to retake the original training.
    b. Future Intervention: Based on the group's highest need, identify the top 3 alternative or future intervention types needed for the group.

**Retake Rule:** Recommend that an employee should retake the **original training plan** if the 'Rate of the intervention’s overall impact to the efficiency of the personnel' is 2 or lower (Unsatisfactory/Poor).
"""
        
        user_message = f"""
You are the Batangas State University HR Data Analyst AI.
Analyze the CATNA data and the impact assessment data provided below and generate a single JSON response adhering strictly to the required Pydantic schema: `GeminiAnalysisResponse`.

# DATA TO ANALYZE

CATNA Analysis Summary (Pre-calculated):
{json.dumps({
    "Overall_Top_3_Needs": overall_top_3(df),
    "Group_Mean_Plans_Office_College": group_mean_plans(df, "Office/College"),
    "Percentage_Breakdowns": percentage_breakdowns(df, "Office/College")
}, indent=2)}

Raw Employee Data (for granular analysis):
{json.dumps(json.loads(data_to_send), indent=2)}

# ANALYSIS INSTRUCTIONS

1. **Key Insights & Prioritization (For key_insights_and_prioritization field):**
   - Highlight the top 3 overall training needs and the lowest-rated components by group.
   - Provide strategic priorities for HRMO based on the mean scores.

2. **Impact Assessment (For impact_assessment_details field and individual_impact_list field):**
   {prompt_impact_instructions}
"""
        
        # --- LLM API CALL (Structured) ---
        ai_response_dict = call_gemini_with_retries_structured(
            user_message, 
            GeminiAnalysisResponse # Pass the Pydantic model for structured output
        )

        # Check for API failure (returns dict with "error" key)
        if "error" in ai_response_dict:
             raise HTTPException(status_code=500, detail=ai_response_dict["error"])
        
        # Parse the structured response
        gemini_struct = GeminiAnalysisResponse(**ai_response_dict)

        # --- FINAL JSON RESPONSE CONSTRUCTION ---
        
        # 1. Prepare map for fast lookup of impact data
        impact_lookup = {}
        if gemini_struct.individual_impact_list:
            for item in gemini_struct.individual_impact_list:
                impact_lookup[item.name] = {
                    "Overall_Impact_Rating": item.overall_impact_rating,
                    "Retake_Decision": item.retake_decision
                }

        # 2. **REVERTED MERGE:** We now keep the Individual_Training_Plans list CLEAN.
        final_individual_plans = individual_training_plans(df)
        # We do NOT merge impact data into this list anymore.
        
        # 3. Determine the content for impact_assessment (Group Summary + Individual List)
        impact_details = {}
        if has_impact_data:
             # Include the structured list of individual impact data in the impact_assessment field
             impact_details["Individual_Impact_Retake_Data"] = [item.model_dump() for item in gemini_struct.individual_impact_list]
             # Include the markdown group summary
             if gemini_struct.impact_assessment_details:
                 impact_details["Gemini_Group_Assessment_Details"] = gemini_struct.impact_assessment_details

        # Return combined result
        return JSONResponse(content={
            "catna_analysis_summary": {
                "Overall_Top_3_Needs": overall_top_3(df),
                "Individual_Training_Plans": final_individual_plans, # <--- CLEAN LIST (NO IMPACT DATA)
                "Group_Mean_Plans_Office_College": group_mean_plans(df, "Office/College"),
                "Percentage_Breakdowns": percentage_breakdowns(df, "Office/College")
            },
            # This key now contains BOTH the structured individual data AND the group summary
            "impact_assessment": impact_details, 
            
            # This key receives ONLY the general CATNA insights (Sections 1 & 2)
            "gemini_insights": gemini_struct.key_insights_and_prioritization 
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Server Error during file processing: {e}")
        # Catch JSON decoding errors from LLM or Pydantic validation errors
        raise HTTPException(status_code=500, detail=f"Internal server error: LLM structure failure or processing error: {str(e)}")

    finally:
        # Cleanup
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)