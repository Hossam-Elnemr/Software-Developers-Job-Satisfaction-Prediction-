"""
    python predict_raw_test_data.py --test new_test_data.csv --output predictions.csv
"""
import argparse
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")


# ============================================================
# Helpers
# ============================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_unique_columns(df):
    seen = {}
    new_cols = []

    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}__dup{seen[col]}")

    df = df.copy()
    df.columns = new_cols
    return df


def split_multi_values(series):
    return series.fillna("").astype(str).apply(
        lambda x: [item.strip() for item in x.split(";") if item.strip()]
    )


def transform_multi_hot(df, cols, mlbs, sep):
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            df[col] = ""

        mlb = mlbs.get(col)

        if mlb is None:
            df = df.drop(columns=[col], errors="ignore")
            continue

        lists = split_multi_values(df[col])
        arr = mlb.transform(lists)
        new_cols = [f"{col}{sep}{cls}" for cls in mlb.classes_]

        encoded = pd.DataFrame(arr, columns=new_cols, index=df.index)

        df = pd.concat(
            [df.drop(columns=[col], errors="ignore"), encoded],
            axis=1,
        )
    return df


def simplify_aiexplain(x):
    x = str(x).lower()

    if "yes" in x:
        return "yes"

    if "no" in x:
        return "no"

    return "other"


def group_dev_type(dev_type):
    if pd.isna(dev_type):
        return "Unknown"

    dev_type = str(dev_type)

    if any(k in dev_type for k in ["back-end", "front-end", "full-stack", "mobile", "desktop", "embedded", "game", "Architect", "QA"]):
        return "Software Engineering"

    if any(k in dev_type for k in ["Data scientist", "Data engineer", "AI/ML", "AI apps", "Applied scientist", "Data or business analyst"]):
        return "Data & AI"

    if any(k in dev_type for k in ["Cloud", "DevOps", "System administrator", "Database", "Cybersecurity", "Support"]):
        return "Infrastructure & Ops"

    if any(k in dev_type for k in ["manager", "executive", "Founder", "VP", "Product"]):
        return "Leadership & Product"

    return "Other"


# ============================================================
# Preprocessing blocks
# ============================================================

def preprocess_main_ai(raw_df, prep):
    cfg = prep["main_ai"]
    target = prep["target_col"]

    df = raw_df.reindex(columns=cfg["selected_cols"]).copy()
    df = df.drop(columns=[target], errors="ignore")

    if "WorkExp" in df.columns:
        df["WorkExp"] = pd.to_numeric(df["WorkExp"], errors="coerce")
        df["WorkExp"] = df["WorkExp"].fillna(cfg["workexp_median"])
        df["WorkExp"] = df["WorkExp"].clip(upper=45)

    for col, value in cfg["fill_values"].items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    if "AIExplain" in df.columns:
        df["AIExplain"] = df["AIExplain"].apply(simplify_aiexplain)

    for col in cfg["multi_cols"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df = transform_multi_hot(
        df,
        cols=cfg["multi_cols"],
        mlbs=cfg["mlbs"],
        sep=";",
    )

    encoder = cfg["ohe"]

    if encoder is not None:
        cat_cols = list(getattr(encoder, "feature_names_in_", []))
        cat_cols = [c for c in cat_cols if c in df.columns]

        if cat_cols:
            encoded = pd.DataFrame(
                encoder.transform(df[cat_cols].astype(str)),
                columns=encoder.get_feature_names_out(cat_cols),
                index=df.index,
            )

            df = pd.concat(
                [df.drop(columns=cat_cols, errors="ignore"), encoded],
                axis=1,
            )

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.reindex(columns=cfg["output_columns"], fill_value=0)

    return df


def preprocess_ai_agent(raw_df, prep):
    cfg = prep["ai_agent"]
    target = prep["target_col"]

    df = raw_df.reindex(columns=cfg["cols"]).copy()
    df = df.drop(columns=[target], errors="ignore")

    agree_score = cfg["agree_score"]

    def process_challenge_row(row):
        out = {}

        for col, score in agree_score.items():
            value = row.get(col)

            if pd.isna(value):
                continue

            for challenge in [c.strip() for c in str(value).split(";") if c.strip()]:
                out[challenge] = max(out.get(challenge, 0), score)

        return out

    pivot = pd.DataFrame(df.apply(process_challenge_row, axis=1).tolist(), index=df.index)
    challenge_cols = [c for c in cfg["challenge_cols"] if c in pivot.columns]

    df = df.drop(columns=list(agree_score.keys()), errors="ignore")

    for col in challenge_cols:
        df[col] = pivot[col].fillna(3)

    df["AI_Challenges_Total"] = df[challenge_cols].sum(axis=1) if challenge_cols else 0

    if "AIAgent_Uses" in df.columns:
        def count_uses(x):
            x = str(x).strip()
            if x in ("", "nan", "None"):
                return 0
            return x.count(";") + 1

        df["AIAgent_Uses_Count"] = df["AIAgent_Uses"].apply(count_uses)
        df = df.drop(columns=["AIAgent_Uses"], errors="ignore")

    if "AIHuman" in df.columns:
        h = df["AIHuman"].fillna("").astype(str).str.get_dummies(sep=";")
        h.columns = ["AIHuman_" + c.strip().replace(" ", "_") for c in h.columns]
        h = h.reindex(columns=cfg["aihuman_columns"], fill_value=0)

        df = pd.concat(
            [df.drop(columns=["AIHuman"], errors="ignore"), h],
            axis=1,
        )

    if "YearsCode" in df.columns:
        df["YearsCode"] = pd.to_numeric(df["YearsCode"], errors="coerce")
        df["YearsCode"] = df["YearsCode"].fillna(cfg["years_median"])
        lb, ub = cfg["years_clip"]
        df["YearsCode"] = df["YearsCode"].clip(lb, ub)

    if "OrgSize" in df.columns:
        df["OrgSize"] = df["OrgSize"].fillna("Unknown").map(cfg["org_map"]).fillna(8)

    if "DevType" in df.columns:
        df["DevType_Grouped"] = df["DevType"].apply(group_dev_type)
        df["DevType_encoded"] = df["DevType"].map(cfg["dev_mean"]).fillna(cfg["global_dev_mean"])
        df = df.drop(columns=["DevType", "AIOpen"], errors="ignore")

    if "ICorPM" in df.columns:
        mapping = cfg["icorpm_map"]

        df["ICorPM"] = df["ICorPM"].fillna("Unknown").astype(str)
        fallback = "Unknown" if "Unknown" in mapping else list(mapping.keys())[0]

        df["ICorPM"] = df["ICorPM"].apply(lambda x: x if x in mapping else fallback)
        df["ICorPM"] = df["ICorPM"].map(mapping).fillna(mapping[fallback])

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.reindex(columns=cfg["selected_features"], fill_value=0)

    return df


def preprocess_comp_survey(raw_df, prep):
    cfg = prep["comp_survey"]

    df = raw_df.copy()

    if "Currency" in df.columns:
        df["Clean_Currency"] = df["Currency"].astype(str).str[:3].str.upper()
        df["ExchangeRate"] = df["Clean_Currency"].map(cfg["rates_to_usd"]).fillna(1.0)
    else:
        df["Clean_Currency"] = "NON"
        df["ExchangeRate"] = 1.0

    if "CompTotal" in df.columns:
        df["CompTotal"] = pd.to_numeric(df["CompTotal"], errors="coerce")
        df["CompTotal_USD"] = df["CompTotal"] * df["ExchangeRate"]
    else:
        df["CompTotal_USD"] = np.nan

    if "Country" in df.columns:
        df["Country"] = df["Country"].fillna(cfg["country_mode_train"])

    if "Currency" in df.columns and "Country" in df.columns:
        df["Currency"] = (
            df["Currency"]
            .fillna(df["Country"].map(cfg["country_to_currency_train"]))
            .fillna("USD")
        )

    if "TechEndorseIntro" in df.columns:
        df["TechEndorseIntro"] = df["TechEndorseIntro"].fillna("Skipped")

    for col in cfg["survey_cols_available"]:
        if col not in df.columns:
            df[col] = -1

    if cfg["survey_cols_available"]:
        df[cfg["survey_cols_available"]] = df[cfg["survey_cols_available"]].fillna(-1)

    if cfg["q99_train"] is not None:
        df["CompTotal_USD"] = df["CompTotal_USD"].clip(upper=cfg["q99_train"])

    X = df.reindex(columns=cfg["final_features"]).copy()

    cat_cols = [c for c in cfg["cat_cols"] if c in X.columns]

    if cat_cols and cfg["ordinal_encoder"] is not None:
        X[cat_cols] = cfg["ordinal_encoder"].transform(X[cat_cols].astype(str))

    if cfg["median_imputer"] is not None:
        X = pd.DataFrame(
            cfg["median_imputer"].transform(X),
            columns=cfg["final_features"],
            index=df.index,
        )
    else:
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X


def preprocess_work_database(raw_df, prep):
    cfg = prep["work_database"]
    target = prep["target_col"]

    df = raw_df.reindex(columns=cfg["cols"]).copy()
    df = df.drop(columns=[target], errors="ignore")

    if "LearnCodeChoose" in df.columns:
        df["LearnCodeChoose"] = df["LearnCodeChoose"].fillna(cfg["learn_code_fill"])
        df["LearnCodeChoose_Encoded"] = df["LearnCodeChoose"].apply(
            lambda x: 1 if "Yes, I am new to coding" in str(x) else 0
        )

    if "RemoteWork" in df.columns:
        df["RemoteWork"] = df["RemoteWork"].fillna("Not Specified")
        df["RemoteWork_Encoded"] = df["RemoteWork"].map(cfg["remote_map"]).fillna(2)

    if "Employment" in df.columns:
        df["Employment_Encoded"] = df["Employment"].map(cfg["employment_map"]).fillna(3)

    if "DatabaseAdmired" in df.columns:
        df["DatabaseAdmired"] = df["DatabaseAdmired"].fillna("None")
        df["DB_Admired_Elite"] = df["DatabaseAdmired"].apply(
            lambda x: int(any(db in str(x) for db in cfg["elite_admired"]))
        )

    if "DatabaseHaveWorkedWith" in df.columns:
        df["DatabaseHaveWorkedWith"] = df["DatabaseHaveWorkedWith"].fillna("None")
        df["DB_Worked_Elite"] = df["DatabaseHaveWorkedWith"].apply(
            lambda x: int(any(db in str(x) for db in cfg["elite_worked"]))
        )

    raw_cols = [
        "RemoteWork",
        "Employment",
        "LearnCodeChoose",
        "DatabaseHaveWorkedWith",
        "DatabaseAdmired",
    ]

    X = df.drop(columns=[c for c in raw_cols if c in df.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.reindex(columns=cfg["output_columns"], fill_value=0)

    return X


def preprocess_tools_platforms(raw_df, prep):
    cfg = prep["tools_platforms"]
    target = prep["target_col"]

    df = raw_df.reindex(columns=cfg["cols"]).copy()
    df = df.drop(columns=[target], errors="ignore")
    df = df.drop(columns=cfg["many_nulls"], errors="ignore")

    for col, value in cfg["single_fill_values"].items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    for col, value in cfg["numeric_fill_values"].items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(value)

    for col in cfg["multi_value_cols"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    df = df.drop(columns=cfg["drop_cols"], errors="ignore")

    df = transform_multi_hot(
        df,
        cols=cfg["multi_value_cols"],
        mlbs=cfg["mlbs"],
        sep="_",
    )

    single_cols = [c for c in cfg["single_value_cols"] if c in df.columns]
    df = pd.get_dummies(df, columns=single_cols, drop_first=True, dtype=int)

    X = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.reindex(columns=cfg["columns_after_rare_filter"], fill_value=0)
    X = X.reindex(columns=cfg["selected_features"], fill_value=0)

    return X


def preprocess_stack_overflow(raw_df, prep):
    cfg = prep["stack_overflow"]

    if not cfg["cols_to_test"]:
        return pd.DataFrame(index=raw_df.index)

    df = raw_df.reindex(columns=cfg["cols_to_test"]).copy()

    for col in cfg["cols_to_test"]:
        df[col] = df[col].fillna("Missing").astype(str)

        encoder = cfg["encoders"].get(col)

        if encoder is None:
            df[col] = 0
            continue

        known = set(encoder.classes_)
        fallback = "Missing" if "Missing" in known else encoder.classes_[0]

        df[col] = df[col].apply(lambda x: x if x in known else fallback)
        df[col] = encoder.transform(df[col])

    df = df.drop(columns=cfg["drop_cols"], errors="ignore")
    df = df.reindex(columns=cfg["output_columns"], fill_value=0)

    return df


def preprocess_raw_test_data(raw_df, prep):
    target = prep["target_col"]

    df = raw_df.copy()
    df = df.drop(columns=[target], errors="ignore")
    df = df.drop(columns=prep.get("dropped_high_missing_columns", []), errors="ignore")

    blocks = [
        preprocess_main_ai(df, prep),
        preprocess_ai_agent(df, prep),
        preprocess_comp_survey(df, prep),
        preprocess_work_database(df, prep),
        preprocess_stack_overflow(df, prep),
        preprocess_tools_platforms(df, prep),
    ]

    X = pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)
    X = make_unique_columns(X)

    X = X.reindex(columns=prep["final_feature_columns"], fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    final_scaler = prep.get("scaler")
    X = pd.DataFrame(final_scaler.transform(X), columns = X.columns, index = X.index)
    return X


def normalize_y_true(y_true, preds, prep):
    # If predictions are numeric but y_true is Low/Average/High, map y_true.
    pred_series = pd.Series(preds)

    if pd.api.types.is_numeric_dtype(pred_series) and y_true.dtype == "object":
        mapping = prep.get("target_mapping", {"Low": 0, "Average": 1, "High": 2})
        mapped = y_true.map(mapping)

        if mapped.notna().all():
            return mapped

    return mapped


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Raw test CSV path")
    parser.add_argument("--output", default="predictions.csv", help="Predictions output CSV")
    parser.add_argument("--preprocessing", default="preprocessing_objects/raw_preprocessing.pkl")
    parser.add_argument("--models", default="preprocessing_objects/models.pkl")
    args = parser.parse_args()

    prep = load_pickle(args.preprocessing)
    model_data = load_pickle(args.models)

    models = model_data["models"]
    model_feature_columns = model_data["model_feature_columns"]
    target = prep["target_col"]

    raw_test = pd.read_csv(args.test)

    y_true_raw = raw_test[target].copy()

    X_processed = preprocess_raw_test_data(raw_test, prep)
    
    X_model = X_processed.reindex(columns=model_feature_columns, fill_value=0)


    for i in X_processed.columns:
        X_processed[i].fillna(0)
    
    predictions = pd.DataFrame(index=raw_test.index)
    metrics = []


    for model_name, model in models.items():
        t0 = time.time()
        preds = model.predict(X_model)
        test_time = time.time() - t0

        predictions[f"{model_name}_prediction"] = preds

        
        y_true = normalize_y_true(y_true_raw.copy(), preds, prep)
        print(y_true.dtype)
        acc = (y_true == preds).sum()/len(y_true)
        # bal_acc = balanced_accuracy_score(y_true, preds)
        # macro_f1 = f1_score(y_true, preds, average="macro")

        metrics.append({
            "model": model_name,
            "accuracy": acc,
            # "balanced_accuracy": bal_acc,
            # "macro_f1": macro_f1,
            "test_time_sec": test_time,
        })

        print("=" * 60)
        print(model_name)
        print(f"Accuracy: {acc:.4f}")
        # print(f"Balanced accuracy: {bal_acc:.4f}")
        # print(f"Macro F1: {macro_f1:.4f}")
        print(f"Test time: {test_time:.6f}s")
        # print(classification_report(y_true, preds, zero_division=0))

    output = pd.concat(
        [raw_test.reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    )
    output.to_csv(args.output, index=False)

    print(f"Saved predictions to: {args.output}")

    if metrics:
        metrics_df = pd.DataFrame(metrics)
        print(metrics_df)


if __name__ == "__main__":
    import argparse
    main()
