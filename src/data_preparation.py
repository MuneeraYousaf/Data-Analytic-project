from datasets import load_dataset

# Load the full dataset (all generation methods)
dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

# Inspect available splits
print(dataset)  # should show: by_polishing, from_title, from_title_and_content

# All available splits
splits = ["by_polishing", "from_title", "from_title_and_content"]

# List to store all rows
all_dataframes = []

for split_name in splits:
    df = pd.DataFrame(dataset[split_name])
    
    # Identify AI-generated columns
    ai_cols = [c for c in df.columns if c.endswith("_generated_abstract")]

    # Human-written
    human_df = df[["original_abstract"]].dropna().rename(columns={"original_abstract": "text"})
    human_df["label"] = 0
    human_df["source"] = "human"
    human_df["split"]  = split_name

    # AI-generated
    ai_frames = []
    for c in ai_cols:
        tmp = df[[c]].dropna().rename(columns={c: "text"})
        if not tmp.empty:
            tmp["label"]  = 1
            tmp["source"] = c.replace("_generated_abstract", "")
            tmp["split"]  = split_name
            ai_frames.append(tmp)

    ai_df = pd.concat(ai_frames, ignore_index=True) if ai_frames else pd.DataFrame(columns=["text","label","source","split"])

    # Combine both
    combined_df = pd.concat([human_df, ai_df], ignore_index=True)
    combined_df = combined_df.assign(text=combined_df["text"].astype(str).str.strip())
    combined_df = combined_df[combined_df["text"] != ""].reset_index(drop=True)

    # Add to the main list
    all_dataframes.append(combined_df)
# Merge all splits into one DataFrame
full_df = pd.concat(all_dataframes, ignore_index=True)
print("Combined shape:", full_df.shape)
print(full_df["label"].value_counts())

# Save to file
full_df.to_csv("../data/processed/combined_all.csv", index=False)
print("✅ Saved to: ../data/processed/combined_all.csv")

print("Shape:", full_df.shape)
print("Label counts:\n", full_df["label"].value_counts())
print("Missing text:", full_df["text"].isna().sum())
print("Duplicate rows (text,label):", full_df.duplicated(subset=["text","label"]).sum())
