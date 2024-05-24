import pandas as pd


def create_latex():
    df = pd.read_csv("../temp/4v_500/details_4v_500.csv")
    df = df[df["target_size"] == 5]
    priority_order = ['ghisaconus', 'indian_pines', 'lucas']
    df['dataset'] = pd.Categorical(df['dataset'], categories=priority_order, ordered=True)
    df = df.sort_values('dataset')
    for fold in range(10):
        fold_df = df[df["fold"] == fold]

        g_bands = fold_df[fold_df["dataset"] == "ghisaconus"]["selected_features"].values[0]
        g_bands = g_bands.split("-")
        g_bands = [str(int(band)+1) for band in g_bands]
        g_bands = ", ".join(g_bands)

        i_bands = fold_df[fold_df["dataset"] == "indian_pines"]["selected_features"].values[0]
        i_bands = i_bands.split("-")
        i_bands = [str(int(band)+1) for band in i_bands]
        i_bands = ", ".join(i_bands)

        l_bands = fold_df[fold_df["dataset"] == "lucas"]["selected_features"].values[0]
        l_bands = l_bands.split("-")
        l_bands = [str(int(band)+1) for band in l_bands]
        l_bands = ", ".join(l_bands)

        print(f"{fold+1} & {g_bands} & {i_bands} & {l_bands} \\\\")


create_latex()