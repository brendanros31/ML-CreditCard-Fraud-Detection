import matplotlib.pyplot as plt
import seaborn as sns


# Heatmap and correlation
def heatmap(corr_df, _annot=True, show=True):
    if show==True:
        plt.figure(figsize=(10,5))
        sns.heatmap(corr_df, cmap="coolwarm", annot=_annot)
        plt.show()

