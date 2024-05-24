import pandas as pd
import os


def remove_mcuve():
    summary = 'saved\\1_all_ex_spa\\fsdr.csv'
    details = 'saved\\1_all_ex_spa\\details_fsdr.csv'

    summary_df = pd.read_csv(summary)
    details_df = pd.read_csv(details)

    summary_df = summary_df[ (summary_df["algorithm"] != "mcuve") | (summary_df["dataset"] != "lucas")]
    summary_df.to_csv(summary, index=False)

    details_df = details_df[(details_df["algorithm"] != "mcuve") | (details_df["dataset"] != "lucas")]
    details_df.to_csv(details, index=False)


def remove_mcuve2():
    summary = 'saved\\1_mcuve_spa\\mcuve_luc.csv'
    details = 'saved\\1_mcuve_spa\\details_mcuve_luc.csv'

    summary_df = pd.read_csv(summary)
    details_df = pd.read_csv(details)

    summary_df = summary_df[ (summary_df["algorithm"] != "mcuve") | (summary_df["dataset"] != "lucas_min")]
    summary_df.to_csv(summary, index=False)

    details_df = details_df[(details_df["algorithm"] != "mcuve") | (details_df["dataset"] != "lucas_min")]
    details_df.to_csv(details, index=False)


def remove_lucasmin():
    summary = 'saved\\1_all_ex_spa\\fsdr.csv'
    details = 'saved\\1_all_ex_spa\\details_fsdr.csv'

    summary_df = pd.read_csv(summary)
    details_df = pd.read_csv(details)

    summary_df = summary_df[ (summary_df["dataset"] != "lucas_min")]
    summary_df.to_csv(summary, index=False)

    details_df = details_df[ (details_df["dataset"] != "lucas_min")]
    details_df.to_csv(details, index=False)

def remove_mcuve_lucasmin():
    summary = 'saved\\1_spa_luc\\mcuve_luc.csv'
    details = 'saved\\1_spa_luc\\details_mcuve_luc.csv'

    summary_df = pd.read_csv(summary)
    details_df = pd.read_csv(details)

    summary_df = summary_df[ (summary_df["algorithm"] != "mcuve") | (summary_df["dataset"] != "lucas_min")]
    summary_df.to_csv(summary, index=False)

    details_df = details_df[(details_df["algorithm"] != "mcuve") | (details_df["dataset"] != "lucas_min")]
    details_df.to_csv(details, index=False)


def lasso_remove():
    root = "saved"
    locations = [os.path.join(root, subfolder) for subfolder in os.listdir(root) if subfolder.startswith("1_") and subfolder!= "1_lasso"]

    for loc in locations:
        files = os.listdir(loc)
        for f in files:
            if "all_features_" in f:
                continue
            if "bsdr-" in f:
                continue
            p = os.path.join(loc, f)
            df = pd.read_csv(p)
            df = df[ (df["algorithm"] != "lasso")]
            df.to_csv(p, index=False)

lasso_remove()

