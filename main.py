import uvicorn
import os
import pandas as pd
import json
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
from google import genai 
from dotenv import load_dotenv # <-- NEW IMPORT

# =========================
# Environment & Client Setup
# =========================

# Load environment variables explicitly from config.env
load_dotenv(dotenv_path='config.env')
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    # Use a placeholder client that will immediately fail if the key is missing.
    # The HTTPException in the endpoint will handle the user-facing error.
    print("FATAL WARNING: GEMINI_API_KEY not found. Please check config.env.")
    client = None
else:
    client = genai.Client(api_key=API_KEY)


# =========================
# Training Configuration
# (Kept for local analysis if needed, though this version skips structured Pydantic)
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
def call_gemini_with_retries(user_message, retries=5, delay=5):
    if client is None:
        return "ERROR: Gemini client is not initialized due to missing API Key."
        
    last_error = None
    for attempt in range(retries):
        try:
            # Using gemini-2.5-flash for better quota handling
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_message
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            # Log the error but suppress the retry logic output to the console for cleaner output
            # print(f"Gemini API busy, retry {attempt+1}/{retries}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2 # Exponential backoff
            
    print(f"Gemini API unavailable after {retries} retries. Last error: {last_error}")
    return f"ERROR: Gemini AI is currently unavailable after retries. (Last Error: {last_error})"

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

        # Run PURE DATA analysis (no LLM yet)
        df = load_multiple_excels([tmp_path])
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid data found in Excel file.")
            
        # Ensure key columns exist
        for col in ["Office/College", "Supervisor Name", "Position", "Campus", "Years in Current Position", "Name"]:
            if col not in df.columns:
                df[col] = "N/A"
                
        df = calculate_individual_means(df)

        # Generate structured JSON data from pandas for LLM input
        analysis_json = {
            "Overall_Top_3_Needs": overall_top_3(df),
            "Individual_Training_Plans": individual_training_plans(df),
            "Group_Mean_Plans_Office_College": group_mean_plans(df, "Office/College"),
            "Group_Mean_Plans_Supervisor": group_mean_plans(df, "Supervisor Name"),
            "Group_Mean_Plans_Campus": group_mean_plans(df, "Campus"),
            "Percentage_Breakdowns": percentage_breakdowns(df, "Office/College") +
                                     percentage_breakdowns(df, "Position") +
                                     percentage_breakdowns(df, "Supervisor Name") +
                                     percentage_breakdowns(df, "Campus") +
                                     percentage_breakdowns(df, "Years in Current Position"),
        }
        
        # --- LLM Insight Generation (Unstructured) ---
        user_message = f"""
You are a data analyst AI for Batangas State University. Analyze the CATNA data provided below and extract key insights.

JSON Data for Analysis:
{json.dumps(analysis_json, indent=2)}

Please provide:
1. Key insights about training gaps (referencing components and groups).
2. Recommendations for training prioritization.
3. Notable trends or patterns across groups (Position, Office/College, Campus).
"""
        ai_reply = call_gemini_with_retries(user_message)

        # Check for AI errors (like the quota exhaustion)
        if ai_reply.startswith("ERROR:"):
             raise HTTPException(status_code=500, detail=ai_reply)

        # Return combined result
        return JSONResponse(content={
            "catna_analysis": analysis_json,
            "gemini_insights": ai_reply
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Server Error during file processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {str(e)}")

    finally:
        # Cleanup
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)