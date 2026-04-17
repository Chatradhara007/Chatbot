import pandas as pd
import pymongo
from groq import Groq
import config

_dfs = None


def get_dfs():
    global _dfs
    if _dfs is not None:
        return _dfs

    client = pymongo.MongoClient(config.MONGO_URI)
    db = client["placementsDB"]

    # ── Load base collections ────────────────────────────────────────────────
    _dfs = {
        'batch':   pd.DataFrame(list(db["batch_reports"].find({}, {"_id": 0}))),
        'branch':  pd.DataFrame(list(db["branch_reports"].find({}, {"_id": 0}))),
        'company': pd.DataFrame(list(db["company_records"].find({}, {"_id": 0}))),
    }

    # ── Load company_details to get name → ID mapping ───────────────────────
    details_raw = list(db["company_details"].find({}, {"_id": 0}))
    _dfs['company_details'] = pd.DataFrame(details_raw) if details_raw else pd.DataFrame(
        columns=["companyID", "companyName", "companySector", "companyDesc"]
    )

    # ── Build company_roles: one row per role, WITH companyName ──────────────
    # Merge company_records with company_details so we always have the name
    company_df = _dfs['company']
    details_df = _dfs['company_details']

    # Build a companyID -> companyName lookup
    if not details_df.empty and 'companyID' in details_df.columns and 'companyName' in details_df.columns:
        id_to_name = dict(zip(details_df['companyID'], details_df['companyName']))
    else:
        id_to_name = {}

    rows = []
    for _, row in company_df.iterrows():
        cid = row.get('companyID')
        cname = id_to_name.get(cid, cid)   # fall back to ID only if name truly missing
        roles = row.get('roles') or []
        if isinstance(roles, list) and len(roles) > 0:
            for r in roles:
                ctc_raw = r.get('roleCTC', None)
                # Normalise roleCTC to a float (handles "7 LPA", 7, "7.5", None)
                if ctc_raw is None:
                    ctc_num = None
                else:
                    import re
                    m = re.search(r'[\d.]+', str(ctc_raw))
                    ctc_num = float(m.group()) if m else None

                rows.append({
                    'companyID':     cid,
                    'companyName':   cname,
                    'year':          row.get('year'),
                    'totalOffers':   row.get('totalOffers'),
                    'totalSelected': row.get('totalSelected'),
                    'roleType':      r.get('roleType', 'Unknown'),
                    'roleCTC':       ctc_raw,   # original string kept for display
                    'roleCTC_num':   ctc_num,   # numeric for sorting/comparisons
                })
        else:
            rows.append({
                'companyID':     cid,
                'companyName':   cname,
                'year':          row.get('year'),
                'totalOffers':   row.get('totalOffers'),
                'totalSelected': row.get('totalSelected'),
                'roleType':      None,
                'roleCTC':       None,
                'roleCTC_num':   None,
            })

    _dfs['company_roles'] = pd.DataFrame(rows)
    return _dfs


groq_client = Groq(api_key=config.GROQ_API_KEY)


def get_pandas_code(user_query: str) -> str:
    prompt = f"""You are a Python/pandas code generator for VNR college placements database.

You have a dictionary of Pandas DataFrames called `dfs`. Here are ALL available DataFrames:

1. dfs['batch']
   columns: batch (str, e.g. '2022-26', '2023-27'), total_students (int),
            placed (int), placement_percentage (float),
            avg_package (float, in LPA), highest_package (float, in LPA)
   NOTE: 'batch' is a range like '2023-27' — the last year is the graduation year.

2. dfs['branch']
   columns: branch (str, e.g. 'CSE', 'CSBS', 'ECE', 'EEE', 'MECH', 'CIVIL'),
            onRolls (int), registered (int), eligible (int), notEligible (int),
            placed (int), unplaced (int), multiple (int),
            noOfStudentsMultiple (int), totalOffers (int),
            placementPercent (float), highestSalary (float),
            lowestSalary (float), averageSalary (float)

3. dfs['company']
   columns: companyID (str), totalOffers (int), totalSelected (int), year (str)
   NOTE: companyID is an internal identifier — NEVER show it to the user.

4. dfs['company_details']
   columns: companyID (str), companyName (str), companySector (str), companyDesc (str)

5. dfs['company_roles']   ← USE THIS for ALL salary / package / CTC queries
   columns:
     companyID     (str)   — internal ID, NEVER show to user
     companyName   (str)   — human-readable name, ALWAYS use this for display
     year          (str)
     totalOffers   (int)
     totalSelected (int)
     roleType      (str)   — e.g. "Software Engineer"
     roleCTC       (str)   — original string like "7 LPA" (use for display)
     roleCTC_num   (float) — numeric value already extracted (use for sorting/math)

STRICT RULES:
1. NEVER display companyID in the output. Always use companyName.
2. For highest/lowest/average package: use roleCTC_num (already numeric, no extraction needed).
3. For "highest package company":
   idx = dfs['company_roles']['roleCTC_num'].idxmax()
   row = dfs['company_roles'].loc[idx]
   print(f"{{row['companyName']}} offered the highest package of {{row['roleCTC']}}.")
4. For company name search, use case-insensitive contains on companyName:
   df[df['companyName'].str.lower().str.contains('keyword', na=False)]
5. Always drop NaN before numeric operations:
   df_valid = dfs['company_roles'][dfs['company_roles']['roleCTC_num'].notna()]
6. If a filter returns an empty DataFrame, print: "I don't have that information in the current database."
7. Do NOT import pandas or redefine dfs. Both are already available.
8. Output ONLY executable Python code. No explanation, no markdown, no triple backticks.

Query: {user_query}
"""
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config.GROQ_SMART_MODEL,
        temperature=0.0,
    )
    code = response.choices[0].message.content.strip()

    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])

    return code.strip()


def run_code_safely(code: str, dfs: dict) -> str:
    import io, contextlib
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {"dfs": dfs, "pd": pd})
        result = stdout_capture.getvalue().strip()
        return result if result else "No output was produced."
    except Exception as e:
        return f"Code execution error: {e}"


def get_data_agent():
    dfs = get_dfs()

    def agent(query: str) -> str:
        code = get_pandas_code(query)
        print(f"[Debug] Generated code:\n{code}\n")
        result = run_code_safely(code, dfs)
        return result

    return agent
