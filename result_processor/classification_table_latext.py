import pandas as pd


def create_latex_table(metric, dataset):
    head = r"""
\begin{table}[H]
\caption{Overall Accuracy (OA) based on 10-fold cross validation results for different target sizes on Indian Pines dataset. BSDR outperforms other algorithms with a consistent standard deviation.\label{reg_oa}}
\begin{tabularx}{\textwidth}{Lrrrrrr}
\toprule
\multicolumn{1}{c}{\textbf{}} & \multicolumn{6}{c}{\textbf{Target Size}} \\
\cline{2-7}
\multicolumn{1}{c}{\textbf{Algorithm}} & \textbf{5} & \textbf{10} & \textbf{15} & \textbf{20} & \textbf{25} & \textbf{30}\\
\midrule
            """

    tail = r"""\end{tabularx}
    \noindent{\footnotesize{}}
    \end{table}"""

    time_df = pd.read_csv(f"../final_results/{dataset}_time.csv")
    oa_df = pd.read_csv(f"../final_results/{dataset}_oa.csv")
    kappa_df = pd.read_csv(f"../final_results/{dataset}_kappa.csv")
    
    priority_order = ['MCUVE','SPA','BS-Net-FC','Zhang et al.', 'BSDR','All Bands']
    
    time_df['algorithm'] = pd.Categorical(time_df['algorithm'], categories=priority_order, ordered=True)
    time_df = time_df.sort_values('algorithm')

    oa_df['algorithm'] = pd.Categorical(oa_df['algorithm'], categories=priority_order, ordered=True)
    oa_df = oa_df.sort_values('algorithm')

    kappa_df['algorithm'] = pd.Categorical(kappa_df['algorithm'], categories=priority_order, ordered=True)
    kappa_df = kappa_df.sort_values('algorithm')

    keys = ['time', 'oa', 'kappa']
    dfs = [time_df, oa_df, kappa_df]
    index = keys.index(metric)

    targets_size = [5, 10, 15, 20, 25, 30]
    mid = ""

    base_df = dfs[index]
    for algorithm in priority_order:
        if algorithm == "All Bands":
            continue
        if metric == "time" and algorithm == "All Bands":
            continue
        df = base_df[base_df['algorithm'] == algorithm]
        row = df.iloc[0]

        if algorithm == "All Bands":
            mid = mid + algorithm + r" & \multicolumn{6}{c}{"+ row[f"{metric}_{targets_size[0]}"] + r"} \\" +"\n"
        else:
            mid = mid + f"{algorithm} "
            for t in targets_size:
                key = f"{metric}_{t}"
                mid = mid + f"& {row[key]} "
            mid = mid + r"\\ " + "\n"
    mid = mid + r"\bottomrule " +"\n"

    print(head + mid + tail)

create_latex_table("time","GHISACONUS")

