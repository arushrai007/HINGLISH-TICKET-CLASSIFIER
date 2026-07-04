import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("data/raw_messages.csv")

df["message"]=df["message"].str.strip()
df=df.dropna(subset=["message","label"])
df=df.drop_duplicates(subset=["message"])

print("class distribution:\n", df["label"].value_counts())


#to create label2id and id2label mapping simply label encoding.
label_list = sorted(df["label"].unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
df["label_id"] = df["label"].map(label2id)



train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label_id"], random_state=42 #stratification** ensures each split (train/val/test) has roughly the same proportion of each class, so your validation results aren't skewed by an accidentally imbalanced split
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42
)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


