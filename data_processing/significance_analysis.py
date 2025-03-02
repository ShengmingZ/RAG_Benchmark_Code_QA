import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_perplexity_changed_subset(perplexity_low_recall, perplexity_high_recall,
                                      dataset_name="Dataset", threshold=0.0, alpha=0.05):
    """
    Analyze perplexity differences between low and high recall conditions, focusing on
    the subset of samples where perplexity actually changed.

    Parameters:
    -----------
    perplexity_low_recall : array-like
        Perplexity values under low retrieval recall condition
    perplexity_high_recall : array-like
        Perplexity values under high retrieval recall condition (same samples)
    dataset_name : str
        Name of the dataset for reporting
    threshold : float
        Minimum absolute difference to consider a sample as "changed"
    alpha : float
        Significance level

    Returns:
    --------
    results : dict
        Dictionary containing analysis results for both full dataset and changed subset
    """
    # Ensure arrays are numpy arrays of the same length
    perplexity_low_recall = np.asarray(perplexity_low_recall)
    perplexity_high_recall = np.asarray(perplexity_high_recall)

    if len(perplexity_low_recall) != len(perplexity_high_recall):
        raise ValueError("Both perplexity arrays must have the same length")

    # Calculate differences (low recall - high recall)
    # Positive differences mean perplexity decreased with higher recall
    differences = perplexity_low_recall - perplexity_high_recall

    # ----- ANALYSIS OF FULL DATASET -----
    # Run Wilcoxon signed-rank test on full dataset
    full_stat, full_p = stats.wilcoxon(differences, alternative='greater')

    # Calculate effect size for full dataset
    n_full = len(differences)
    # Safer z-score calculation that handles extreme p-values better
    if full_p < 1e-10:
        z_full = 6.5  # Approximate z-score for extremely small p-values
    else:
        z_full = stats.norm.ppf(1 - full_p / 2)  # two-tailed p to z
    effect_size_full = z_full / np.sqrt(n_full)

    # ----- IDENTIFY CHANGED SUBSET -----
    # Find samples where perplexity changed beyond the threshold
    changed_indices = np.where(np.abs(differences) > threshold)[0]
    changed_differences = differences[changed_indices]

    # Calculate percentage of samples that changed
    n_changed = len(changed_indices)
    percent_changed = (n_changed / n_full) * 100

    # ----- ANALYSIS OF CHANGED SUBSET -----
    subset_results = {}

    if n_changed > 0:
        # Run Wilcoxon signed-rank test on changed subset
        subset_stat, subset_p = stats.wilcoxon(changed_differences, alternative='greater')

        # Calculate effect size for changed subset
        if subset_p < 1e-10:
            z_subset = 6.5  # Approximate z-score for extremely small p-values
        else:
            z_subset = stats.norm.ppf(1 - subset_p / 2)  # two-tailed p to z
        effect_size_subset = z_subset / np.sqrt(n_changed)

        # Interpret effect size
        def interpret_effect_size(effect):
            if abs(effect) < 0.1:
                return "negligible"
            elif abs(effect) < 0.3:
                return "small"
            elif abs(effect) < 0.5:
                return "medium"
            else:
                return "large"

        # Calculate improved/worse percentages within changed subset
        improved_subset = np.sum(changed_differences > 0)
        worse_subset = np.sum(changed_differences < 0)

        subset_results = {
            'sample_size': n_changed,
            'wilcoxon_statistic': subset_stat,
            'p_value': subset_p,
            'significant': subset_p < alpha,
            'effect_size': effect_size_subset,
            'effect_interpretation': interpret_effect_size(effect_size_subset),
            'mean_difference': np.mean(changed_differences),
            'median_difference': np.median(changed_differences),
            'std_difference': np.std(changed_differences),
            'improved_count': improved_subset,
            'worse_count': worse_subset,
            'improved_percent': (improved_subset / n_changed) * 100 if n_changed > 0 else 0,
            'worse_percent': (worse_subset / n_changed) * 100 if n_changed > 0 else 0,
        }

    # ----- COMBINED RESULTS -----
    # Calculate descriptive statistics for full dataset
    improved_full = np.sum(differences > 0)
    same_full = np.sum(np.abs(differences) <= threshold)
    worse_full = np.sum(differences < -threshold)

    results = {
        'dataset': dataset_name,
        'threshold': threshold,

        # Full dataset results
        'full': {
            'sample_size': n_full,
            'wilcoxon_statistic': full_stat,
            'p_value': full_p,
            'significant': full_p < alpha,
            'effect_size': effect_size_full,
            'effect_interpretation': interpret_effect_size(effect_size_full),
            'mean_difference': np.mean(differences),
            'median_difference': np.median(differences),
            'std_difference': np.std(differences),
            'improved_count': improved_full,
            'same_count': same_full,
            'worse_count': worse_full,
            'improved_percent': improved_full / n_full * 100,
            'same_percent': same_full / n_full * 100,
            'worse_percent': worse_full / n_full * 100,
        },

        # Changed subset metadata
        'percent_samples_changed': percent_changed,

        # Changed subset results (if any samples changed)
        'changed_subset': subset_results if n_changed > 0 else {
            'sample_size': 0,
            'note': 'No samples changed beyond the threshold'
        }
    }

    return results


def visualize_subset_analysis(results, show_full_data=True):
    """
    Visualize the results of the changed subset analysis.

    Parameters:
    -----------
    results : dict
        Results from analyze_perplexity_changed_subset
    show_full_data : bool
        Whether to include visualizations of the full dataset

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    dataset_name = results['dataset']
    threshold = results['threshold']

    # Determine number of plots based on whether changed subset exists
    has_changed_subset = 'sample_size' in results['changed_subset'] and results['changed_subset']['sample_size'] > 0

    if has_changed_subset:
        if show_full_data:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes = [None, None, axes[0], axes[1]]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = [axes[0], axes[1], None, None]

    # Plot histogram and distribution of full dataset
    if show_full_data:
        # full_differences = results['full']['differences']

        # Histogram of all differences
        # sns.histplot(full_differences, kde=True, ax=axes[0])
        axes[0].axvline(x=0, color='red', linestyle='--')
        axes[0].set_xlabel('Perplexity Difference (Low Recall - High Recall)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Full {dataset_name} (n={results["full"]["sample_size"]})')

        # Add threshold lines if applicable
        if threshold > 0:
            axes[0].axvline(x=threshold, color='green', linestyle='--', alpha=0.7,
                            label=f'Threshold (±{threshold})')
            axes[0].axvline(x=-threshold, color='green', linestyle='--', alpha=0.7)
            axes[0].legend()

        # Add statistics annotation
        full_stats_text = (
            f"Full Dataset:\n"
            f"p-value: {results['full']['p_value']:.4f}\n"
            f"Effect size: {results['full']['effect_size']:.3f} "
            f"({results['full']['effect_interpretation']})\n\n"
            f"Improved: {results['full']['improved_percent']:.1f}%\n"
            f"Same: {results['full']['same_percent']:.1f}%\n"
            f"Worse: {results['full']['worse_percent']:.1f}%"
        )

        axes[0].annotate(full_stats_text, xy=(0.03, 0.97), xycoords='axes fraction',
                         ha='left', va='top',
                         bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Pie chart showing percentage of changed samples
        if has_changed_subset:
            changed = results['percent_samples_changed']
            unchanged = 100 - changed

            axes[1].pie([changed, unchanged], labels=['Changed', 'Unchanged'],
                        autopct='%1.1f%%', startangle=90,
                        colors=['#ff9999', '#66b3ff'])
            axes[1].set_title(f'Percentage of Samples Changed\n(threshold = {threshold})')
        else:
            axes[1].text(0.5, 0.5, f"No samples changed\nbeyond threshold {threshold}",
                         ha='center', va='center', fontsize=14)
            axes[1].axis('off')

    # Plot histogram and stats for changed subset
    if has_changed_subset:
        # subset_differences = results['changed_subset']['differences']

        # Histogram of differences in changed subset
        if axes[2] is not None:
            # sns.histplot(subset_differences, kde=True, ax=axes[2])
            axes[2].axvline(x=0, color='red', linestyle='--')
            axes[2].set_xlabel('Perplexity Difference (Low Recall - High Recall)')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title(f'Changed Subset (n={results["changed_subset"]["sample_size"]})')

            # Add statistics annotation
            subset_stats_text = (
                f"Changed Subset:\n"
                f"p-value: {results['changed_subset']['p_value']:.4f}\n"
                f"Effect size: {results['changed_subset']['effect_size']:.3f} "
                f"({results['changed_subset']['effect_interpretation']})\n\n"
                f"Improved: {results['changed_subset']['improved_percent']:.1f}%\n"
                f"Worse: {results['changed_subset']['worse_percent']:.1f}%"
            )

            axes[2].annotate(subset_stats_text, xy=(0.03, 0.97), xycoords='axes fraction',
                             ha='left', va='top',
                             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Direction comparison (improved vs. worsened)
        if axes[3] is not None:
            improved = results['changed_subset']['improved_count']
            worse = results['changed_subset']['worse_count']

            axes[3].bar(['Improved', 'Worsened'], [improved, worse])
            axes[3].set_title('Direction of Change in Subset')
            axes[3].set_ylabel('Number of Samples')

            # Add percentage labels
            for i, v in enumerate([improved, worse]):
                axes[3].text(i, v + 0.5, f"{v} ({v / (improved + worse) * 100:.1f}%)",
                             ha='center', va='bottom')

    plt.tight_layout()
    return fig


def compare_datasets_subset_analysis(results1, results2):
    """
    Compare the results of subset analysis between two datasets.

    Parameters:
    -----------
    results1 : dict
        Results from analyze_perplexity_changed_subset for dataset 1
    results2 : dict
        Results from analyze_perplexity_changed_subset for dataset 2

    Returns:
    --------
    comparison : dict
        Dictionary with comparison results
    """
    comparison = {
        'dataset1': results1['dataset'],
        'dataset2': results2['dataset'],
        'full_sample_size_ratio': results1['full']['sample_size'] / results2['full']['sample_size'],
        'percent_changed': {
            'dataset1': results1['percent_samples_changed'],
            'dataset2': results2['percent_samples_changed'],
            'ratio': results1['percent_samples_changed'] / results2['percent_samples_changed']
            if results2['percent_samples_changed'] > 0 else float('inf')
        }
    }

    # Compare changed subsets if both exist
    has_subset1 = 'sample_size' in results1['changed_subset'] and results1['changed_subset']['sample_size'] > 0
    has_subset2 = 'sample_size' in results2['changed_subset'] and results2['changed_subset']['sample_size'] > 0

    if has_subset1 and has_subset2:
        subset_comparison = {
            'sample_size_ratio': results1['changed_subset']['sample_size'] / results2['changed_subset']['sample_size'],
            'effect_size_ratio': results1['changed_subset']['effect_size'] / results2['changed_subset']['effect_size']
            if results2['changed_subset']['effect_size'] != 0 else float('inf'),
            'mean_difference_ratio': results1['changed_subset']['mean_difference'] / results2['changed_subset'][
                'mean_difference']
            if results2['changed_subset']['mean_difference'] != 0 else float('inf'),
            'improved_percent_ratio': results1['changed_subset']['improved_percent'] / results2['changed_subset'][
                'improved_percent']
            if results2['changed_subset']['improved_percent'] != 0 else float('inf'),
            'both_significant': results1['changed_subset']['significant'] and results2['changed_subset']['significant']
        }
        comparison['changed_subset'] = subset_comparison
    else:
        comparison['changed_subset'] = {'note': 'Cannot compare - one or both datasets have no changed samples'}

    return comparison


if __name__ == '__main__':
    data_list = [[1.2581325138002153, 1.06955767976151, 1.1799598601999453, 1.1021221975696704, 1.1007394932268135, 1.084039158092242, 1.085050263474326, 1.2046522911706352, 1.1869185615731004, 1.1176260591434126, 1.1007314323781285, 1.049970467873029, 1.3083921222196597, 1.0937347160143482, 1.1456015891217455, 1.1872400547710467, 1.1667485414204646, 1.1244870489245902, 1.1052520216341597, 1.0661000490169823, 1.114006334870804, 1.1036769143248046, 1.2886324645395542, 1.0530532128696122, 1.0530935793655989, 1.136873780938784, 1.0714020080185604, 1.3159954886799796, 1.2274706087063079, 1.1582836204086566, 1.0508136930259524, 1.0736800036453615, 1.0416768394809317, 1.189633006639218, 1.1048885834982243, 1.1710710582928234, 1.123637708593896, 1.1173145860646094, 1.0743667399988048, 1.1439301331121254, 1.1829967113771025, 1.1286786074356363, 1.0741428466313132, 1.125380677994181, 1.0979422035533875, 1.1461804754689293, 1.155139357256428, 1.184370400992144, 1.1888986674022821, 1.1294710561358416, 1.284382239952294, 1.2521843105889672, 1.0724385139630421, 1.0954385490852805, 1.1312869123134535, 1.1524617437471572, 1.1209562908965534, 1.1171791967715436, 1.0685173129940544, 1.060655529660689, 1.1267790874436703, 1.1330080133670941, 1.1229777504099665, 1.197639790232391, 1.1002360060824168, 1.1340776890180038, 1.2259641462868296, 1.0915991070742044, 1.0545751770554996, 1.2090106059017471, 1.0552581352262826, 1.104608573734826, 1.1011505300735571, 1.1930650511637215, 1.088263863617319, 1.1011505300735571, 1.062010426147078, 1.1966894350545307, 1.0892615724925097, 1.0559820571677507, 1.1518592629764701, 1.0895718785690565, 1.0870510469907244, 1.1455451989855243],
[1.1909202196173954, 1.0594508242065444, 1.1718559441300629, 1.2131659141206954, 1.0635329402524087, 1.050902726019494, 1.050902726019494, 1.144530366289747, 1.1424351701022903, 1.0661959454595793, 1.0883729185516717, 1.1331019830155087, 1.1597050711019026, 1.099484553983093, 1.1416110719364632, 1.0798717252636105, 1.1640414006497366, 1.130451179560741, 1.1466648926038907, 1.0553733338143085, 1.0776721555127244, 1.0746825617509945, 1.0719312747300314, 1.0499462221869522, 1.0495124206633244, 1.1450090677040288, 1.051062206576467, 1.318464266962154, 1.1472325101129326, 1.0377008179716756, 1.077265280896909, 1.1069445911607685, 1.0582585186010023, 1.1950179015425053, 1.02552157978808, 1.0614159681028092, 1.0925228896562922, 1.2129783895596453, 1.1008579671392102, 1.1440705265638802, 1.1438768715449532, 1.1162276300895067, 1.0588914403096286, 1.1605054492342002, 1.1388866145930092, 1.125895139868598, 1.1179311118389563, 1.1078669612322056, 1.107257901151625, 1.0481335516081838, 1.1483693549742717, 1.2425513564803548, 1.2183472331448661, 1.0656315321016954, 1.0997757673841286, 1.118662029385967, 1.06773542355167, 1.0945557795561942, 1.0635705544904515, 1.0548053765243837, 1.0940563000151113, 1.1140378302525737, 1.1351259411673098, 1.120263539504489, 1.1375595865342392, 1.1251555268224351, 1.1221360930820854, 1.0284579548064436, 1.0705082644910222, 1.3506963231872824, 1.060611241029233, 1.0643899028358623, 1.1184452535373017, 1.099654713205119, 1.123054492977857, 1.1184452535373017, 1.0686399904761383, 1.2629127200282588, 1.0412161510790818, 1.1962860052723776, 1.4176643385742629, 1.0851913261202741, 1.1143492031859679, 1.164207001439214]
]

    data_list = [[1.0174375063763244, 1.0563444927075325, 1.0095994654742515, 1.0686691847815062, 1.0147752165163424, 1.032351832779185, 1.044837762053463, 1.0122701697179257, 1.0135319099069158, 1.0208319420848757, 1.02203469445299, 1.0543234351125934, 1.1932006043780725, 1.0427723827340176, 1.0777312462857858, 1.0987688843984458, 1.0445472799312414, 1.0472990853641342, 1.0145796315855007, 1.022800790317249, 1.0017181130144535, 1.1167490727276002, 1.0456021662718158, 1.0026384248877354, 1.002210043661637, 1.144384619083331, 1.0247566305186726, 1.0663541914039156, 1.0417847188041507, 1.0322942126570527, 1.0299518000377605, 1.0351857388015624, 1.0348598142645606, 1.0731277117400024, 1.0036596435026233, 1.0282318886271908, 1.078335136174076, 1.0029925345695458, 1.0821983508158532, 1.0427216401606567, 1.06086977685532, 1.0068153175963144, 1.0717806053148247, 1.0491456195085171, 1.0497566336447308, 1.051619168492518, 1.1439712743420027, 1.0363119970036547, 1.0457050516983906, 1.0623315494680639, 1.052884340361019, 1.1097899172759553, 1.0562573696693613, 1.0815761302141293, 1.0679921398013683, 1.038122464492435, 1.0706029636829768, 1.0069017275571304, 1.0223716928697104, 1.0551768394614573, 1.010851371074323, 1.0651104505203266, 1.0604949007796687, 1.0136714319569609, 1.013463163825304, 1.076855436758955, 1.053287427415323, 1.001555331810521, 1.0163065293167017, 1.0815647331611964, 1.0014919829145539, 1.0244112603070967, 1.0006249452030147, 1.0006685001572475, 1.051864748369198, 1.0006206711106587, 1.0129040833609282, 1.0684695871522847, 1.0170414919428536, 1.0777960532498414, 1.0387513542219504, 1.0038014263877888, 1.1339429297957933, 1.0644979233385394],
[1.0150593568155848, 1.027189285343415, 1.0116571161688588, 1.0368335385629053, 1.0060262461147897, 1.010489601751729, 1.021072147504854, 1.0112925325220616, 1.0120745621261207, 1.0128846662617763, 1.095872036521258, 1.0440428643491584, 1.0054000847189242, 1.0050665612831367, 1.0615534632862869, 1.0416607711212535, 1.022824878701113, 1.0304818128236184, 1.024728553653352, 1.0038271242810868, 1.0416056142655314, 1.000578499316572, 1.0010905190135795, 1.015943242991724, 1.0099027119886148, 1.0114965043932305, 1.0013073955478267, 1.0493792830681568, 1.038764905865267, 1.0055412683266498, 1.0415756813174677, 1.041600016056422, 1.0111494093447762, 1.012559404028144, 1.002220254520375, 1.0184067071554508, 1.0436146913740385, 1.0594943905021723, 1.034185115780023, 1.0738240324805173, 1.0151192088015986, 1.0398624181538507, 1.0049590521468907, 1.0123487591902536, 1.0398642113475414, 1.064484664469261, 1.0775196638590916, 1.018274832314804, 1.0130189652935517, 1.0028982160249449, 1.049700431308075, 1.1101094732194616, 1.0460614684897491, 1.0091279609488455, 1.0576772288399399, 1.012209303689225, 1.1101680926767274, 1.1180730645835844, 1.0032456920130128, 1.0093152383784985, 1.084873645458005, 1.0700058455893362, 1.0823881953881893, 1.012698239462059, 1.0359356151548245, 1.0754015848575196, 1.0214327296262407, 1.0295841935924248, 1.0142487893612944, 1.1023350905676208, 1.0237393663392607, 1.01077821383788, 1.0007421794303042, 1.0031732045464936, 1.0424838784881547, 1.0007273374537524, 1.0299715467285457, 1.031569600439039, 1.10324267334484, 1.0328097398105531, 1.0496426757551802, 1.0401822374634566, 1.1365269846787756, 1.0963554406856117]
]

    # sample_indices = np.random.choice(2000, 200, replace=False)
    #
    # # Sample datasets at those indices
    # sampled_datasets = [
    #     [dataset[idx] for idx in sample_indices]
    #     for dataset in data_list
    # ]
    # data_list = sampled_datasets


    # Example usage:
    # Define a reasonable threshold based on expected perplexity changes
    threshold = 0.0001  # Only consider samples where perplexity changed by at least 0.5

    # Dataset 1 (2000 samples)
    results = []
    for idx in range(len(data_list)-1):
        results.append(analyze_perplexity_changed_subset(
            data_list[idx], data_list[idx+1],
            dataset_name="Dataset 1 (2000 samples)",
            threshold=threshold
        ))


    # Visualize results
    fig1 = visualize_subset_analysis(results[0])
    plt.savefig('dataset1_subset_analysis.png', dpi=300, bbox_inches='tight')

    for idx, result in enumerate(results):
        print(f"{idx * 0.2} -> {(idx + 1) * 0.2}")
        print('  p-value: ', result['full']['p_value'])
        print('  significant: ', result['full']['significant'])
        print('  effect_size: ', result['changed_subset']['effect_size'])
        print('  improved_percent: ', result['changed_subset']['improved_percent'])
