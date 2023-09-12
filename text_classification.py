import pandas as pd
from sklearn.model_selection import train_test_split

# import data
df = pd.read_table("", names=["Datapoint"])
df["Datapoint"] = df["Datapoint"].str.lower()
df[["dialog_act", "utterance_content"]] = df["Datapoint"].str.split(pat=" ", n=1, expand=True)
df.drop("Datapoint", axis=1, inplace=True)
# sanity check TODO remove when assignment is done
print(df.head())

# create train- and test set
train_df, test_df = train_test_split(df, test_size=0.15)
