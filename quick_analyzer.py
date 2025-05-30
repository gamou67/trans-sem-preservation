#!/usr/bin/env python3
"""
Analyseur rapide du repository existant
Version simplifiée pour exécution immédiate
"""

import os
import sys
import ast
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

class QuickRepositoryAnalyzer:
    """Analyseur rapide et simple du repository"""
    
    def __init__(self):
        self.repo_url = "https://github.com/elbachir67/trans-neur-emb-sem-pre-meta-trans.git"
        self.repo_name = "trans-neur-emb-sem-pre-meta-trans"
        self.analysis_results = {}
    
    def clone_or_update_repo(self) -> bool:
        """Clone le repository ou le met à jour s'il existe"""
        print("🔄 Vérification/clonage du repository...")
        
        if os.path.exists(self.repo_name):
            print(f"✅ Repository {self.repo_name} trouvé localement")
            
            # Vérifier si c'est un repo git
            if os.path.exists(os.path.join(self.repo_name, '.git')):
                print("🔄 Mise à jour du repository...")
                try:
                    os.chdir(self.repo_name)
                    subprocess.run(['git', 'pull'], check=True, capture_output=True)
                    print("✅ Repository mis à jour")
                    os.chdir('..')
                    return True
                except subprocess.CalledProcessError:
                    print("⚠️  Impossible de mettre à jour - continuons avec la version locale")
                    os.chdir('..')
                    return True
                except FileNotFoundError:
                    print("⚠️  Git non trouvé - continuons avec la version locale")
                    return True
            else:
                print("⚠️  Répertoire existe mais n'est pas un repo Git")
                return True
        else:
            print(f"📥 Clonage du repository depuis {self.repo_url}")
            try:
                subprocess.run(['git', 'clone', self.repo_url], check=True)
                print("✅ Repository cloné avec succès")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Erreur lors du clonage: {e}")
                print("💡 Vérifiez votre connexion internet et que Git est installé")
                return False
            except FileNotFoundError:
                print("❌ Git non trouvé. Veuillez installer Git pour cloner le repository")
                print("💡 Ou téléchargez manuellement le repository depuis GitHub")
                return False
    
    def analyze_repository(self) -> Dict[str, Any]:
        """Analyse le repository"""
        repo_path = Path(self.repo_name)
        
        if not repo_path.exists():
            print(f"❌ Repository non trouvé: {repo_path}")
            return {}
        
        print(f"🔍 Analyse du repository: {repo_path}")
        
        analysis = {
            'repository_path': str(repo_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'files': {
                'total': 0,
                'python': [],
                'other': []
            },
            'code_metrics': {
                'total_lines': 0,
                'python_files_count': 0
            },
            'python_structure': {
                'classes': [],
                'functions': [],
                'imports': set()
            },
            'key_components': {
                'token_pair_related': [],
                'embedding_related': [],
                'evaluation_related': [],
                'model_related': []
            },
            'errors': [],
            'recommendations': []
        }
        
        # Scanner tous les fichiers
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                analysis['files']['total'] += 1
                
                if file_path.suffix == '.py':
                    rel_path = str(file_path.relative_to(repo_path))
                    analysis['files']['python'].append(rel_path)
                    analysis['code_metrics']['python_files_count'] += 1
                    
                    # Analyser le fichier Python
                    self._analyze_python_file(file_path, repo_path, analysis)
                else:
                    analysis['files']['other'].append(str(file_path.relative_to(repo_path)))
        
        # Convertir les sets en listes pour JSON
        analysis['python_structure']['imports'] = list(analysis['python_structure']['imports'])
        
        # Générer des recommandations
        self._generate_recommendations(analysis)
        
        self.analysis_results = analysis
        return analysis
    
    def _analyze_python_file(self, file_path: Path, repo_path: Path, analysis: Dict):
        """Analyse un fichier Python spécifique"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = len(content.split('\n'))
                analysis['code_metrics']['total_lines'] += lines
                
                # Parser l'AST
                try:
                    tree = ast.parse(content)
                    self._extract_ast_info(tree, file_path, repo_path, analysis)
                except SyntaxError as e:
                    error_msg = f"Erreur syntaxe dans {file_path.relative_to(repo_path)}: {e}"
                    analysis['errors'].append(error_msg)
                    
        except Exception as e:
            error_msg = f"Erreur lecture {file_path.relative_to(repo_path)}: {e}"
            analysis['errors'].append(error_msg)
    
    def _extract_ast_info(self, tree: ast.AST, file_path: Path, repo_path: Path, analysis: Dict):
        """Extrait les informations de l'AST"""
        rel_file_path = str(file_path.relative_to(repo_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'file': rel_file_path,
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    'line_count': getattr(node, 'end_lineno', 1) - getattr(node, 'lineno', 1)
                }
                analysis['python_structure']['classes'].append(class_info)
                
                # Catégoriser les classes importantes
                class_name_lower = node.name.lower()
                if 'token' in class_name_lower or 'pair' in class_name_lower:
                    analysis['key_components']['token_pair_related'].append(class_info)
                elif 'embed' in class_name_lower:
                    analysis['key_components']['embedding_related'].append(class_info)
                elif 'eval' in class_name_lower or 'assess' in class_name_lower:
                    analysis['key_components']['evaluation_related'].append(class_info)
                elif 'model' in class_name_lower:
                    analysis['key_components']['model_related'].append(class_info)
            
            elif isinstance(node, ast.FunctionDef):
                # Fonctions au niveau module (pas dans une classe)
                function_info = {
                    'name': node.name,
                    'file': rel_file_path,
                    'args': [arg.arg for arg in node.args.args],
                    'line_count': getattr(node, 'end_lineno', 1) - getattr(node, 'lineno', 1)
                }
                analysis['python_structure']['functions'].append(function_info)
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis['python_structure']['imports'].add(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis['python_structure']['imports'].add(node.module)
    
    def _generate_recommendations(self, analysis: Dict):
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        
        # Évaluer la complexité
        total_classes = len(analysis['python_structure']['classes'])
        total_functions = len(analysis['python_structure']['functions'])
        
        if total_classes > 10:
            recommendations.append("🟢 Bon nombre de classes détectées - Code bien structuré")
        elif total_classes > 0:
            recommendations.append("🟡 Structure modulaire de base présente")
        else:
            recommendations.append("🔴 Peu de classes trouvées - Code potentiellement procédural")
        
        # Évaluer les composants clés
        key_components = analysis['key_components']
        if key_components['token_pair_related']:
            recommendations.append("✅ Composants TokenPair trouvés - Réutilisation possible")
        if key_components['embedding_related']:
            recommendations.append("✅ Composants Embedding trouvés - Base solide pour extension")
        if key_components['evaluation_related']:
            recommendations.append("✅ Composants d'évaluation trouvés - Métriques existantes")
        
        # Évaluer les dépendances
        imports = analysis['python_structure']['imports']
        ml_deps = [imp for imp in imports if any(keyword in imp.lower() 
                  for keyword in ['torch', 'transform', 'sklearn', 'numpy'])]
        if ml_deps:
            recommendations.append("✅ Dépendances ML détectées - Compatible avec notre approche")
        
        # Évaluer la qualité
        if len(analysis['errors']) == 0:
            recommendations.append("✅ Code sans erreurs de parsing - Qualité élevée")
        elif len(analysis['errors']) < 5:
            recommendations.append("⚠️  Quelques erreurs mineures détectées")
        else:
            recommendations.append("🔴 Nombreuses erreurs détectées - Code nécessitant attention")
        
        # Recommandations d'intégration
        if (len(key_components['token_pair_related']) > 0 and 
            len(key_components['embedding_related']) > 0):
            recommendations.append("🎯 Stratégie recommandée: Extension par héritage")
        elif total_classes > 5:
            recommendations.append("🎯 Stratégie recommandée: Adaptation avec adaptateurs")
        else:
            recommendations.append("🎯 Stratégie recommandée: Implémentation propre avec inspiration")
        
        analysis['recommendations'] = recommendations
    
    def print_analysis_report(self):
        """Affiche le rapport d'analyse"""
        if not self.analysis_results:
            print("❌ Aucune analyse disponible")
            return
        
        analysis = self.analysis_results
        
        print("\n" + "="*70)
        print("📊 RAPPORT D'ANALYSE DU REPOSITORY")
        print("="*70)
        
        print(f"📁 Repository: {analysis['repository_path']}")
        print(f"🕒 Analysé le: {analysis['timestamp']}")
        print(f"📄 Fichiers totaux: {analysis['files']['total']}")
        print(f"🐍 Fichiers Python: {analysis['code_metrics']['python_files_count']}")
        print(f"📝 Lignes de code: {analysis['code_metrics']['total_lines']:,}")
        
        print(f"\n🏗️  STRUCTURE PYTHON")
        print(f"   Classes: {len(analysis['python_structure']['classes'])}")
        print(f"   Fonctions: {len(analysis['python_structure']['functions'])}")
        print(f"   Imports uniques: {len(analysis['python_structure']['imports'])}")
        
        print(f"\n🎯 COMPOSANTS CLÉS IDENTIFIÉS")
        key_components = analysis['key_components']
        print(f"   TokenPair related: {len(key_components['token_pair_related'])}")
        print(f"   Embedding related: {len(key_components['embedding_related'])}")
        print(f"   Evaluation related: {len(key_components['evaluation_related'])}")
        print(f"   Model related: {len(key_components['model_related'])}")
        
        # Afficher les classes importantes
        if key_components['token_pair_related']:
            print(f"\n🔑 CLASSES TOKEN PAIR")
            for cls in key_components['token_pair_related'][:5]:
                print(f"   • {cls['name']} ({len(cls['methods'])} méthodes) - {cls['file']}")
        
        if key_components['embedding_related']:
            print(f"\n🧠 CLASSES EMBEDDING")
            for cls in key_components['embedding_related'][:5]:
                print(f"   • {cls['name']} ({len(cls['methods'])} méthodes) - {cls['file']}")
        
        # Dépendances importantes
        important_imports = [imp for imp in analysis['python_structure']['imports'] 
                           if any(keyword in imp.lower() 
                                 for keyword in ['torch', 'transform', 'sklearn', 'numpy', 'pandas'])]
        if important_imports:
            print(f"\n📦 DÉPENDANCES IMPORTANTES")
            for imp in sorted(important_imports)[:10]:
                print(f"   • {imp}")
        
        # Erreurs
        if analysis['errors']:
            print(f"\n⚠️  ERREURS DÉTECTÉES ({len(analysis['errors'])})")
            for error in analysis['errors'][:3]:
                print(f"   • {error}")
            if len(analysis['errors']) > 3:
                print(f"   ... et {len(analysis['errors']) - 3} autres")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS")
        for rec in analysis['recommendations']:
            print(f"   {rec}")
        
        print("="*70)
    
    def save_analysis(self, filename: str = "repository_analysis.json"):
        """Sauvegarde l'analyse en JSON"""
        if not self.analysis_results:
            print("❌ Aucune analyse à sauvegarder")
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Analyse sauvegardée: {filename}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
    
    def run_complete_analysis(self):
        """Exécute l'analyse complète"""
        print("🚀 Analyseur Rapide - Framework Préservation Sémantique")
        print("="*60)
        
        start_time = time.time()
        
        # Étape 1: Cloner/vérifier le repository
        if not self.clone_or_update_repo():
            print("❌ Impossible d'obtenir le repository")
            return False
        
        # Étape 2: Analyser
        print(f"\n🔍 Analyse en cours...")
        analysis = self.analyze_repository()
        
        if not analysis:
            print("❌ Échec de l'analyse")
            return False
        
        duration = time.time() - start_time
        
        # Étape 3: Afficher les résultats
        self.print_analysis_report()
        
        print(f"\n⏱️  Durée totale: {duration:.2f} secondes")
        
        # Étape 4: Sauvegarder
        self.save_analysis()
        
        # Étape 5: Prochaines étapes
        print(f"\n🚀 PROCHAINES ÉTAPES RECOMMANDÉES")
        print("   1. Examiner les classes TokenPair et Embedding identifiées")
        print("   2. Tester l'importation des modules trouvés")
        print("   3. Créer l'adaptateur legacy basé sur cette analyse")
        print("   4. Implémenter les classes enrichies avec héritage")
        
        return True

def main():
    """Point d'entrée principal"""
    analyzer = QuickRepositoryAnalyzer()
    
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✅ Analyse terminée avec succès!")
        print("📄 Consultez le fichier 'repository_analysis.json' pour plus de détails")
    else:
        print("\n❌ Analyse échouée")
        print("💡 Vérifiez votre connexion internet et que Git est installé")

if __name__ == "__main__":
    main()