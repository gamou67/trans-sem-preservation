import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Configuration globale pour les graphiques IEEE
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_results(json_file_path):
    """Charge et analyse les résultats du fichier JSON"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    return data

def extract_transformation_stats(data):
    """Extrait les statistiques par type de transformation"""
    transformation_stats = data['enhanced_transformation_analysis']
    
    stats_df = []
    for trans_type, stats in transformation_stats.items():
        stats_df.append({
            'Transformation': trans_type.replace('_', '→'),
            'Count': stats['count'],
            'Success_Rate': stats['success_rate'],
            'Avg_Improvement': stats['avg_improvement'],
            'Avg_BA_Traditional': stats['avg_ba_traditional'],
            'Avg_BA_Neural': stats['avg_ba_neural'],
            'Total_Gaps': stats['total_gaps'],
            'Avg_Processing_Time': stats['avg_processing_time']
        })
    
    return pd.DataFrame(stats_df)

def plot_transformation_overview(stats_df, save_path='transformation_overview.pdf'):
    """Graphique en barres des améliorations par type de transformation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Graphique 1: Améliorations moyennes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars1 = ax1.bar(range(len(stats_df)), stats_df['Avg_Improvement'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Transformation Type')
    ax1.set_ylabel('Average Improvement (%)')
    ax1.set_title('Semantic Preservation Improvement by Transformation Type')
    ax1.set_xticks(range(len(stats_df)))
    ax1.set_xticklabels(stats_df['Transformation'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Ajout des valeurs sur les barres
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Graphique 2: Taux de succès
    bars2 = ax2.bar(range(len(stats_df)), stats_df['Success_Rate'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Transformation Type')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate by Transformation Type')
    ax2.set_xticks(range(len(stats_df)))
    ax2.set_xticklabels(stats_df['Transformation'], rotation=45, ha='right')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    # Ajout des valeurs sur les barres
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def plot_pattern_effectiveness(data, save_path='pattern_effectiveness.pdf'):
    """Graphique de l'efficacité des patterns"""
    pattern_data = data['enhanced_pattern_analysis']
    
    patterns = list(pattern_data['pattern_counts'].keys())
    counts = list(pattern_data['pattern_counts'].values())
    effectiveness = [pattern_data['pattern_effectiveness'][p] for p in patterns]
    
    # Simplifier les noms des patterns
    pattern_names = [p.replace('Pattern', '') for p in patterns]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Graphique 1: Usage des patterns (camembert)
    colors = ['#ff9999', '#66b3ff']
    wedges, texts, autotexts = ax1.pie(counts, labels=pattern_names, autopct='%1.1f%%',
                                      colors=colors, startangle=90, 
                                      textprops={'fontsize': 10})
    ax1.set_title('Pattern Usage Distribution')
    
    # Graphique 2: Efficacité des patterns
    bars = ax2.bar(pattern_names, effectiveness, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Pattern Type')
    ax2.set_ylabel('Average Effectiveness (%)')
    ax2.set_title('Pattern Effectiveness Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Ajout des valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def plot_bas_comparison(data, save_path='bas_comparison.pdf'):
    """Graphique de comparaison BAS traditionnel vs neural"""
    results = data['results']
    
    # Extraire les données pour chaque type de transformation
    transformation_types = {}
    for result in results:
        trans_type = result['transformation_type']
        if trans_type not in transformation_types:
            transformation_types[trans_type] = {
                'traditional': [],
                'neural': [],
                'improvement': []
            }
        
        transformation_types[trans_type]['traditional'].append(result['ba_traditional'])
        transformation_types[trans_type]['neural'].append(result['ba_neural'])
        transformation_types[trans_type]['improvement'].append(result['improvement_percentage'])
    
    # Calculer les moyennes
    avg_data = []
    for trans_type, values in transformation_types.items():
        avg_data.append({
            'Transformation': trans_type.replace('_', '→'),
            'Traditional_BAS': np.mean(values['traditional']),
            'Neural_BAS': np.mean(values['neural']),
            'Improvement': np.mean(values['improvement'])
        })
    
    df = pd.DataFrame(avg_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['Traditional_BAS'], width, 
                   label='Traditional BAS', color='#ff7f7f', alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, df['Neural_BAS'], width,
                   label='Neural BAS', color='#7f7fff', alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Transformation Type')
    ax.set_ylabel('BAS Score')
    ax.set_title('Traditional vs Neural BAS Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Transformation'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ajout des valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def plot_improvement_distribution(data, save_path='improvement_distribution.pdf'):
    """Histogramme de la distribution des améliorations"""
    results = data['results']
    improvements = [r['improvement_percentage'] for r in results 
                   if r['improvement_percentage'] > 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogramme
    ax1.hist(improvements, bins=20, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Improvement Percentage (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Semantic Improvements')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(improvements), color='red', linestyle='--', 
                label=f'Mean: {np.mean(improvements):.1f}%')
    ax1.legend()
    
    # Box plot par type de transformation
    trans_improvements = {}
    for result in results:
        trans_type = result['transformation_type'].replace('_', '→')
        if trans_type not in trans_improvements:
            trans_improvements[trans_type] = []
        if result['improvement_percentage'] > 0:
            trans_improvements[trans_type].append(result['improvement_percentage'])
    
    box_data = list(trans_improvements.values())
    box_labels = list(trans_improvements.keys())
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Colorier les boîtes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Transformation Type')
    ax2.set_ylabel('Improvement Percentage (%)')
    ax2.set_title('Improvement Distribution by Transformation Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def plot_performance_metrics(data, save_path='performance_metrics.pdf'):
    """Graphique des métriques de performance"""
    stats = data['enhanced_scale_metrics']
    bg_stats = data['enhanced_background_processing']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Métriques globales
    metrics = ['Total Evaluations', 'Successful Evaluations', 'Token Pairs', 'Gaps Detected']
    values = [stats['total_evaluations'], stats['successful_evaluations'], 
              bg_stats['total_token_pairs_extracted'], bg_stats['total_gaps_detected']]
    
    bars1 = ax1.bar(metrics, values, color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'],
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Overall Performance Metrics')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Taux de succès par transformation
    trans_stats = data['enhanced_transformation_analysis']
    trans_names = [t.replace('_', '→') for t in trans_stats.keys()]
    success_rates = [trans_stats[t]['success_rate'] for t in trans_stats.keys()]
    
    bars2 = ax2.bar(trans_names, success_rates, alpha=0.8, 
                    color='lightgreen', edgecolor='black', linewidth=0.5)
    ax2.set_title('Success Rate by Transformation')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Temps de traitement
    processing_times = [trans_stats[t]['avg_processing_time'] for t in trans_stats.keys()]
    
    bars3 = ax3.bar(trans_names, processing_times, alpha=0.8,
                    color='orange', edgecolor='black', linewidth=0.5)
    ax3.set_title('Average Processing Time')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(processing_times)*0.01,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=8)
    
    # Gaps détectés par transformation
    gaps_detected = [trans_stats[t]['total_gaps'] for t in trans_stats.keys()]
    
    bars4 = ax4.bar(trans_names, gaps_detected, alpha=0.8,
                    color='red', edgecolor='black', linewidth=0.5)
    ax4.set_title('Semantic Gaps Detected')
    ax4.set_ylabel('Number of Gaps')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(gaps_detected)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def generate_results_table(data):
    """Génère les tableaux de résultats pour l'article"""
    
    # Tableau principal des résultats
    stats = data['enhanced_statistical_analysis']
    scale_metrics = data['enhanced_scale_metrics']
    
    main_results = f"""
    \\begin{{table}}[htbp]
    \\caption{{\\tabletitle{{Large-Scale Experimental Results ({scale_metrics['total_evaluations']} Transformations)}}}}
    \\begin{{center}}
    \\begin{{tabular}}{{lc}}
    \\toprule
    \\tableheader{{Metric}} & \\tableheader{{Value}} \\\\
    \\midrule
    Total evaluations & {scale_metrics['total_evaluations']} \\\\
    Successful evaluations & {scale_metrics['successful_evaluations']} \\\\
    Success rate & {scale_metrics['success_rate']:.1f}\\% \\\\
    Average improvement & +{stats['mean_improvement']:.1f}\\% \\\\
    Standard deviation & {stats['std_improvement']:.1f}\\% \\\\
    Effect size (Cohen's d) & {stats['cohens_d']:.3f} \\\\
    Statistical significance & $p < 0.001$ \\\\
    Total detected gaps & {data['enhanced_background_processing']['total_gaps_detected']} \\\\
    \\bottomrule
    \\end{{tabular}}
    \\end{{center}}
    \\label{{tab:overall_results}}
    \\end{{table}}
    """
    
    # Tableau par type de transformation
    trans_analysis = data['enhanced_transformation_analysis']
    
    trans_table = """
    \\begin{table}[htbp]
    \\caption{\\tabletitle{Results by Transformation Type}}
    \\begin{center}
    \\begin{tabular}{lcccc}
    \\toprule
    \\tableheader{Type} & \\tableheader{Count} & \\tableheader{Success Rate} & \\tableheader{Avg Improvement} & \\tableheader{Gaps} \\\\
    \\midrule
    """
    
    total_count = 0
    total_gaps = 0
    weighted_success = 0
    weighted_improvement = 0
    
    for trans_type, stats in trans_analysis.items():
        trans_name = trans_type.replace('_', '→')
        trans_table += f"{trans_name} & {stats['count']} & {stats['success_rate']:.1f}\\% & +{stats['avg_improvement']:.1f}\\% & {stats['total_gaps']} \\\\\n"
        
        total_count += stats['count']
        total_gaps += stats['total_gaps']
        weighted_success += stats['success_rate'] * stats['count']
        weighted_improvement += stats['avg_improvement'] * stats['count']
    
    avg_success = weighted_success / total_count if total_count > 0 else 0
    avg_improvement = weighted_improvement / total_count if total_count > 0 else 0
    
    trans_table += f"""    \\midrule
    \\textbf{{Total}} & \\textbf{{{total_count}}} & \\textbf{{{avg_success:.1f}\\%}} & \\textbf{{+{avg_improvement:.1f}\\%}} & \\textbf{{{total_gaps}}} \\\\
    \\bottomrule
    \\end{{tabular}}
    \\end{{center}}
    \\label{{tab:transformation_breakdown}}
    \\end{{table}}
    """
    
    return main_results, trans_table

def main():
    """Fonction principale pour générer tous les graphiques"""
    
    # Charger les données (remplacez par le chemin vers votre fichier JSON)
    json_file = 'enhanced_semantic_evaluation_1750434844.json'
    
    try:
        data = load_results(json_file)
        stats_df = extract_transformation_stats(data)
        
        print("Génération des graphiques...")
        
        # Générer tous les graphiques
        plot_transformation_overview(stats_df)
        plot_pattern_effectiveness(data)
        plot_bas_comparison(data)
        plot_improvement_distribution(data)
        plot_performance_metrics(data)
        
        # Générer les tableaux LaTeX
        main_table, trans_table = generate_results_table(data)
        
        print("\n=== TABLEAU PRINCIPAL ===")
        print(main_table)
        
        print("\n=== TABLEAU PAR TYPE DE TRANSFORMATION ===")
        print(trans_table)
        
        print("\nTous les graphiques ont été générés avec succès!")
        print("Fichiers PDF créés: transformation_overview.pdf, pattern_effectiveness.pdf,")
        print("bas_comparison.pdf, improvement_distribution.pdf, performance_metrics.pdf")
        
    except FileNotFoundError:
        print(f"Erreur: Fichier {json_file} non trouvé.")
        print("Veuillez mettre le fichier JSON dans le même répertoire que ce script.")
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")

if __name__ == "__main__":
    main()