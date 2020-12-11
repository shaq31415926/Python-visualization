import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_missing(missing_stats):
        """Histogram of missing value for every features"""
        plt.style.use('seaborn-white')
        plt.figure(figsize=(7, 5))
        plt.hist(missing_stats['missing_fraction'], bins=np.linspace(0, 1, 11), edgecolor='k', color='red',
                 linewidth=1.5)
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlabel('Missing Fraction', size=14)
        plt.ylabel('Count of Features', size=14)
        plt.title("Fraction of Missing Values Histogram", size=16)

        plt.show()
        
def identify_missing(data, missing_threshold):
    """Find the features with a fraction of missing values above `missing_threshold`"""
    missing_series = data.isnull().sum() / data.shape[0]
    missing_stats = pd.DataFrame(
            missing_series).rename(columns={'index': 'feature',
                                                 0 : 'missing_fraction'})

    # Sort with highest number of missing values on top
    missing_stats = missing_stats.sort_values('missing_fraction', ascending=False)

    # Find the columns with a missing percentage above the threshold
    record_missing = pd.DataFrame(
        missing_series[missing_series > missing_threshold]).reset_index().rename(
        columns={'index': 'feature',
                 0: 'missing_fraction'})
    
    plot_missing(missing_stats)

    to_drop = list(record_missing['feature'])
    print(f'{len(to_drop)} feature(s with greater than {missing_threshold}% missing values\n')