#!/usr/bin/env python3
"""
Framework de Patterns de Correction Automatique - Version Corrigée
Implémentation complète et fonctionnelle des patterns concrets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import re
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Configuration Streamlit
st.set_page_config(
    page_title="Patterns Framework",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pattern-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .pattern-failed {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .improvement-badge {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin: 0.2rem;
    }
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CLASSES DE BASE
# ============================================================================

class PatternType(Enum):
    ANNOTATION = "annotation"
    STRUCTURAL_DECOMPOSITION = "structural_decomposition"
    BEHAVIORAL_ENCODING = "behavioral_encoding"

@dataclass
class Gap:
    """Gap sémantique à corriger"""
    source_name: str
    source_type: str
    target_name: Optional[str]
    target_type: Optional[str]
    similarity: float
    severity: float
    gap_type: str
    properties_lost: Dict[str, Any] = field(default_factory=dict)
    constraints_lost: List[str] = field(default_factory=list)
    relationships_lost: List[str] = field(default_factory=list)

@dataclass
class PatternApplication:
    """Résultat de l'application d'un pattern"""
    pattern_name: str
    pattern_type: PatternType
    source_element: str
    target_element: Optional[str]
    success: bool
    improvement_score: float
    complexity_added: float
    generated_code: str
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# PATTERNS ABSTRAITS
# ============================================================================

class SemanticPattern(ABC):
    """Classe abstraite pour les patterns"""
    
    def __init__(self, name: str, pattern_type: PatternType):
        self.name = name
        self.pattern_type = pattern_type
        self.success_rate = 0.0
        self.applications_count = 0
    
    @abstractmethod
    def is_applicable(self, gap: Gap, target_metamodel: str) -> bool:
        pass
    
    @abstractmethod
    def estimate_improvement(self, gap: Gap) -> float:
        pass
    
    @abstractmethod
    def estimate_complexity(self, gap: Gap) -> float:
        pass
    
    @abstractmethod
    def apply_pattern(self, gap: Gap, target_model: str, target_metamodel: str) -> PatternApplication:
        pass
    
    def calculate_priority(self, gap: Gap) -> float:
        """Calcule la priorité d'application"""
        improvement = self.estimate_improvement(gap)
        complexity = self.estimate_complexity(gap)
        return improvement / max(complexity, 0.1)

# ============================================================================
# PATTERNS CONCRETS
# ============================================================================

class AnnotationPattern(SemanticPattern):
    """Pattern pour préserver métadonnées via annotations"""
    
    def __init__(self):
        super().__init__("AnnotationPattern", PatternType.ANNOTATION)
    
    def is_applicable(self, gap: Gap, target_metamodel: str) -> bool:
        return (
            gap.gap_type in ['metadata', 'constraint'] and
            (gap.properties_lost or gap.constraints_lost) and
            target_metamodel.upper() in ['ECORE', 'JAVA']
        )
    
    def estimate_improvement(self, gap: Gap) -> float:
        lost_items = len(gap.properties_lost) + len(gap.constraints_lost)
        base = min(lost_items * 0.12, 0.5)
        if gap.target_name:
            base *= 1.2
        return min(base, 0.7)
    
    def estimate_complexity(self, gap: Gap) -> float:
        return 0.1 + len(gap.properties_lost) * 0.02 + len(gap.constraints_lost) * 0.03
    
    def apply_pattern(self, gap: Gap, target_model: str, target_metamodel: str) -> PatternApplication:
        if target_metamodel.upper() == "ECORE":
            return self._apply_ecore_annotation(gap, target_model)
        elif target_metamodel.upper() == "JAVA":
            return self._apply_java_annotation(gap, target_model)
        else:
            return self._create_failed_application(gap, f"Métamodèle {target_metamodel} non supporté")
    
    def _apply_ecore_annotation(self, gap: Gap, target_model: str) -> PatternApplication:
        try:
            generated_code = target_model
            improvement = 0.0
            
            # Générer des EAnnotations
            annotations = []
            
            for prop_name, prop_value in gap.properties_lost.items():
                annotation = f'''    <eAnnotations source="semantic_preservation">
        <details key="{prop_name}" value="{prop_value}"/>
    </eAnnotations>'''
                annotations.append(annotation)
                improvement += 0.12
            
            for constraint in gap.constraints_lost:
                clean_constraint = constraint.replace('"', '&quot;')
                annotation = f'''    <eAnnotations source="semantic_preservation/constraints">
        <details key="constraint" value="{clean_constraint}"/>
    </eAnnotations>'''
                annotations.append(annotation)
                improvement += 0.15
            
            if gap.target_name:
                # Insérer dans élément existant
                pattern = rf'<e\w+\s+name="{re.escape(gap.target_name)}"[^>]*>'
                match = re.search(pattern, target_model)
                
                if match:
                    insert_pos = match.end()
                    annotations_str = '\n' + '\n'.join(annotations)
                    generated_code = target_model[:insert_pos] + annotations_str + target_model[insert_pos:]
                    explanation = f"Ajout de {len(annotations)} EAnnotations à {gap.target_name}"
                else:
                    explanation = f"Élément {gap.target_name} non trouvé"
                    improvement = 0.0
            else:
                # Créer nouvel élément
                new_element = f'''
<eClass name="{gap.source_name}_preserved">
    <eAnnotations source="semantic_preservation/lost_element">
        <details key="original_name" value="{gap.source_name}"/>
    </eAnnotations>{''.join(annotations)}
</eClass>'''
                
                package_close = target_model.rfind('</ecore:EPackage>')
                if package_close != -1:
                    generated_code = target_model[:package_close] + new_element + '\n' + target_model[package_close:]
                else:
                    generated_code = target_model + new_element
                
                improvement += 0.2
                explanation = f"Création d'EClass avec annotations pour {gap.source_name}"
            
            return PatternApplication(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                source_element=gap.source_name,
                target_element=gap.target_name,
                success=improvement > 0,
                improvement_score=improvement,
                complexity_added=self.estimate_complexity(gap),
                generated_code=generated_code,
                explanation=explanation,
                metadata={'annotations_added': len(annotations)}
            )
            
        except Exception as e:
            return self._create_failed_application(gap, f"Erreur: {str(e)}")
    
    def _apply_java_annotation(self, gap: Gap, target_model: str) -> PatternApplication:
        try:
            generated_code = target_model
            improvement = 0.0
            
            if gap.target_name:
                # Trouver l'élément Java
                patterns = [
                    rf'((?:public|private|protected)\s+(?:static\s+)?(?:class|interface)\s+{re.escape(gap.target_name)})',
                    rf'((?:public|private|protected)\s+[\w\[\]]+\s+{re.escape(gap.target_name)})',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, target_model, re.MULTILINE)
                    if match:
                        # Générer annotation Java
                        props = ', '.join([f'"{k}={v}"' for k, v in gap.properties_lost.items()])
                        constraints = ', '.join([f'"{c}"' for c in gap.constraints_lost])
                        
                        annotation_parts = []
                        if props:
                            annotation_parts.append(f'properties = {{{props}}}')
                        if constraints:
                            annotation_parts.append(f'constraints = {{{constraints}}}')
                        
                        annotation_content = ', '.join(annotation_parts) if annotation_parts else 'value = "preserved"'
                        java_annotation = f'@SemanticPreservation({annotation_content})\n'
                        
                        insert_pos = match.start()
                        generated_code = target_model[:insert_pos] + java_annotation + target_model[insert_pos:]
                        
                        improvement = len(gap.properties_lost) * 0.1 + len(gap.constraints_lost) * 0.15
                        explanation = f"Ajout annotation @SemanticPreservation à {gap.target_name}"
                        break
                else:
                    explanation = f"Élément {gap.target_name} non trouvé"
            else:
                # Créer classe utilitaire
                utility_class = f'''
@SemanticPreservation(
    originalName = "{gap.source_name}",
    originalType = "{gap.source_type}"
)
public class {gap.source_name}Preserved {{
    public static final String ORIGINAL_NAME = "{gap.source_name}";
}}
'''
                generated_code = target_model + utility_class
                improvement = 0.3
                explanation = f"Création classe utilitaire pour {gap.source_name}"
            
            return PatternApplication(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                source_element=gap.source_name,
                target_element=gap.target_name,
                success=improvement > 0,
                improvement_score=improvement,
                complexity_added=self.estimate_complexity(gap),
                generated_code=generated_code,
                explanation=explanation,
                metadata={'annotation_type': 'SemanticPreservation'}
            )
            
        except Exception as e:
            return self._create_failed_application(gap, f"Erreur Java: {str(e)}")
    
    def _create_failed_application(self, gap: Gap, error_msg: str) -> PatternApplication:
        return PatternApplication(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            source_element=gap.source_name,
            target_element=gap.target_name,
            success=False,
            improvement_score=0.0,
            complexity_added=0.0,
            generated_code="",
            explanation=error_msg
        )

class StructuralDecompositionPattern(SemanticPattern):
    """Pattern pour décomposer éléments complexes"""
    
    def __init__(self):
        super().__init__("StructuralDecompositionPattern", PatternType.STRUCTURAL_DECOMPOSITION)
    
    def is_applicable(self, gap: Gap, target_metamodel: str) -> bool:
        return (
            gap.gap_type in ['structural', 'behavioral'] and
            gap.severity > 0.5 and
            (gap.properties_lost or gap.relationships_lost)
        )
    
    def estimate_improvement(self, gap: Gap) -> float:
        base = gap.severity * 0.5
        if gap.relationships_lost:
            base += len(gap.relationships_lost) * 0.1
        return min(base, 0.7)
    
    def estimate_complexity(self, gap: Gap) -> float:
        return 0.3 + len(gap.properties_lost) * 0.05 + len(gap.relationships_lost) * 0.1
    
    def apply_pattern(self, gap: Gap, target_model: str, target_metamodel: str) -> PatternApplication:
        if target_metamodel.upper() == "ECORE":
            return self._apply_ecore_decomposition(gap, target_model)
        elif target_metamodel.upper() == "JAVA":
            return self._apply_java_decomposition(gap, target_model)
        else:
            return self._create_failed_application(gap, f"Décomposition non supportée pour {target_metamodel}")
    
    def _apply_ecore_decomposition(self, gap: Gap, target_model: str) -> PatternApplication:
        try:
            improvement = 0.0
            elements_created = []
            
            # Créer EClass principale
            main_class = f'''
<eClass name="{gap.source_name}">
    <eAnnotations source="structural_decomposition">
        <details key="original_element" value="{gap.source_name}"/>
    </eAnnotations>'''
            
            # Ajouter EAttributes pour propriétés
            for prop_name, prop_value in gap.properties_lost.items():
                prop_type = self._infer_ecore_type(prop_value)
                main_class += f'''
    <eAttribute name="{prop_name}" eType="{prop_type}"/>'''
                improvement += 0.1
            
            main_class += '\n</eClass>'
            elements_created.append(gap.source_name)
            
            # Insérer dans modèle
            package_close = target_model.rfind('</ecore:EPackage>')
            if package_close != -1:
                generated_code = target_model[:package_close] + main_class + '\n' + target_model[package_close:]
            else:
                generated_code = target_model + main_class
            
            explanation = f"Décomposition de {gap.source_name} avec {len(gap.properties_lost)} propriétés"
            
            return PatternApplication(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                source_element=gap.source_name,
                target_element=gap.target_name,
                success=improvement > 0,
                improvement_score=improvement,
                complexity_added=self.estimate_complexity(gap),
                generated_code=generated_code,
                explanation=explanation,
                metadata={'elements_created': elements_created}
            )
            
        except Exception as e:
            return self._create_failed_application(gap, f"Erreur décomposition: {str(e)}")
    
    def _apply_java_decomposition(self, gap: Gap, target_model: str) -> PatternApplication:
        try:
            improvement = 0.0
            
            # Créer classe Java
            main_class = f'''
@SemanticPreservation(originalName = "{gap.source_name}")
public class {gap.source_name} {{'''
            
            # Ajouter champs pour propriétés
            for prop_name, prop_value in gap.properties_lost.items():
                java_type = self._infer_java_type(prop_value)
                main_class += f'''
    private {java_type} {prop_name};'''
                improvement += 0.1
            
            # Ajouter constructeur et getters
            main_class += f'''
    
    public {gap.source_name}() {{
        // Constructeur
    }}
    
    public String getOriginalName() {{
        return "{gap.source_name}";
    }}
}}
'''
            
            generated_code = target_model + '\n\n' + main_class
            explanation = f"Décomposition Java de {gap.source_name} avec {len(gap.properties_lost)} propriétés"
            
            return PatternApplication(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                source_element=gap.source_name,
                target_element=gap.target_name,
                success=improvement > 0,
                improvement_score=improvement,
                complexity_added=self.estimate_complexity(gap),
                generated_code=generated_code,
                explanation=explanation,
                metadata={'class_created': gap.source_name}
            )
            
        except Exception as e:
            return self._create_failed_application(gap, f"Erreur décomposition Java: {str(e)}")
    
    def _infer_ecore_type(self, value: Any) -> str:
        if isinstance(value, bool):
            return "ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EBoolean"
        elif isinstance(value, int):
            return "ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EInt"
        elif isinstance(value, float):
            return "ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EDouble"
        else:
            return "ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString"
    
    def _infer_java_type(self, value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "double"
        else:
            return "String"
    
    def _create_failed_application(self, gap: Gap, error_msg: str) -> PatternApplication:
        return PatternApplication(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            source_element=gap.source_name,
            target_element=gap.target_name,
            success=False,
            improvement_score=0.0,
            complexity_added=0.0,
            generated_code="",
            explanation=error_msg
        )

class BehavioralEncodingPattern(SemanticPattern):
    """Pattern pour encoder le comportement"""
    
    def __init__(self):
        super().__init__("BehavioralEncodingPattern", PatternType.BEHAVIORAL_ENCODING)
    
    def is_applicable(self, gap: Gap, target_metamodel: str) -> bool:
        return (
            gap.gap_type == 'behavioral' and
            gap.source_type.lower() in ['operation', 'method', 'eoperation'] and
            gap.severity > 0.4
        )
    
    def estimate_improvement(self, gap: Gap) -> float:
        base = gap.severity * 0.4
        if gap.constraints_lost:
            base += len(gap.constraints_lost) * 0.1
        return min(base, 0.6)
    
    def estimate_complexity(self, gap: Gap) -> float:
        return 0.4 + len(gap.constraints_lost) * 0.05
    
    def apply_pattern(self, gap: Gap, target_model: str, target_metamodel: str) -> PatternApplication:
        if target_metamodel.upper() == "ECORE":
            return self._apply_ecore_behavioral(gap, target_model)
        elif target_metamodel.upper() == "JAVA":
            return self._apply_java_behavioral(gap, target_model)
        else:
            return self._create_failed_application(gap, f"Encodage non supporté pour {target_metamodel}")
    
    def _apply_ecore_behavioral(self, gap: Gap, target_model: str) -> PatternApplication:
        try:
            improvement = 0.0
            
            # Créer EClass pour comportement
            behavior_class = f'''
<eClass name="{gap.source_name}Behavior">
    <eAnnotations source="behavioral_encoding">
        <details key="original_operation" value="{gap.source_name}"/>
    </eAnnotations>
    <eAttribute name="operationName" eType="String"/>'''
            
            # Encoder contraintes
            for i, constraint in enumerate(gap.constraints_lost):
                clean_constraint = constraint.replace('"', '&quot;')
                behavior_class += f'''
    <eAttribute name="constraint_{i}" eType="String">
        <eAnnotations source="behavior_constraint">
            <details key="expression" value="{clean_constraint}"/>
        </eAnnotations>
    </eAttribute>'''
                improvement += 0.12
            
            behavior_class += '''
    <eOperation name="executeBehavior" eType="String"/>
</eClass>'''
            
            improvement += 0.15  # Bonus pour opération
            
            # Insérer dans modèle
            package_close = target_model.rfind('</ecore:EPackage>')
            if package_close != -1:
                generated_code = target_model[:package_close] + behavior_class + '\n' + target_model[package_close:]
            else:
                generated_code = target_model + behavior_class
            
            explanation = f"Encodage comportemental de {gap.source_name} avec {len(gap.constraints_lost)} contraintes"
            
            return PatternApplication(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                source_element=gap.source_name,
                target_element=gap.target_name,
                success=improvement > 0,
                improvement_score=improvement,
                complexity_added=self.estimate_complexity(gap),
                generated_code=generated_code,
                explanation=explanation,
                metadata={'behavior_class': f"{gap.source_name}Behavior"}
            )
            
        except Exception as e:
            return self._create_failed_application(gap, f"Erreur encodage: {str(e)}")
    
    def _apply_java_behavioral(self, gap: Gap, target_model: str) -> PatternApplication:
        try:
            improvement = 0.0
            
            # Créer classe d'encodage Java
            behavior_class = f'''
@SemanticPreservation(originalOperation = "{gap.source_name}")
public class {gap.source_name}BehaviorEncoder {{
    private final String operationName = "{gap.source_name}";
    
    public String executeBehavior() {{
        return "Executing: " + operationName;
    }}
    
    public boolean validateConstraints() {{
        return true; // Validation des contraintes
    }}
}}
'''
            
            improvement = 0.2 + len(gap.constraints_lost) * 0.1
            
            generated_code = target_model + '\n\n' + behavior_class
            explanation = f"Encodage comportemental Java de {gap.source_name}"
            
            return PatternApplication(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                source_element=gap.source_name,
                target_element=gap.target_name,
                success=improvement > 0,
                improvement_score=improvement,
                complexity_added=self.estimate_complexity(gap),
                generated_code=generated_code,
                explanation=explanation,
                metadata={'encoder_class': f"{gap.source_name}BehaviorEncoder"}
            )
            
        except Exception as e:
            return self._create_failed_application(gap, f"Erreur encodage Java: {str(e)}")
    
    def _create_failed_application(self, gap: Gap, error_msg: str) -> PatternApplication:
        return PatternApplication(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            source_element=gap.source_name,
            target_element=gap.target_name,
            success=False,
            improvement_score=0.0,
            complexity_added=0.0,
            generated_code="",
            explanation=error_msg
        )

# ============================================================================
# MOTEUR D'APPLICATION
# ============================================================================

class PatternEngine:
    """Moteur d'application des patterns"""
    
    def __init__(self):
        self.patterns = [
            AnnotationPattern(),
            StructuralDecompositionPattern(),
            BehavioralEncodingPattern()
        ]
    
    def apply_patterns(self, gaps: List[Gap], target_model: str, target_metamodel: str) -> Dict[str, Any]:
        applications = []
        current_model = target_model
        total_improvement = 0.0
        total_complexity = 0.0
        successful = 0
        
        for gap in gaps:
            best_pattern = None
            best_priority = 0.0
            
            # Trouver le meilleur pattern pour ce gap
            for pattern in self.patterns:
                if pattern.is_applicable(gap, target_metamodel):
                    priority = pattern.calculate_priority(gap)
                    if priority > best_priority:
                        best_priority = priority
                        best_pattern = pattern
            
            if best_pattern:
                application = best_pattern.apply_pattern(gap, current_model, target_metamodel)
                applications.append(application)
                
                if application.success:
                    current_model = application.generated_code
                    total_improvement += application.improvement_score
                    total_complexity += application.complexity_added
                    successful += 1
        
        return {
            'applications': applications,
            'final_model': current_model,
            'total_improvement': min(total_improvement, 1.0),
            'total_complexity': total_complexity,
            'successful_applications': successful,
            'total_applications': len(applications),
            'success_rate': successful / max(len(applications), 1)
        }

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def create_improvement_chart(applications: List[PatternApplication]) -> go.Figure:
    """Graphique des améliorations par pattern"""
    
    if not applications:
        return go.Figure()
    
    pattern_names = [app.pattern_name.replace("Pattern", "") for app in applications]
    improvements = [app.improvement_score * 100 for app in applications]
    complexities = [app.complexity_added * 100 for app in applications]
    colors = ['green' if app.success else 'red' for app in applications]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Amélioration (%)',
        x=pattern_names,
        y=improvements,
        marker_color=colors,
        opacity=0.7,
        text=[f'+{imp:.1f}%' for imp in improvements],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Améliorations par Pattern Appliqué",
        xaxis_title="Patterns",
        yaxis_title="Amélioration (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    """Interface principale"""
    
    st.markdown('<h1 class="main-header">🎨 Patterns de Correction Automatique</h1>', unsafe_allow_html=True)
    st.markdown("**Framework complet pour corriger automatiquement les gaps sémantiques**")
    
    # Initialiser le moteur
    if 'pattern_engine' not in st.session_state:
        st.session_state.pattern_engine = PatternEngine()
        st.session_state.gaps_data = None
        st.session_state.results = None
    
    engine = st.session_state.pattern_engine
    
    # Sidebar
    with st.sidebar:
        st.header("🎨 Configuration")
        
        target_metamodel = st.selectbox("Métamodèle cible", ["Ecore", "Java"], index=0)
        
        st.subheader("📊 Patterns Disponibles")
        for pattern in engine.patterns:
            st.write(f"• **{pattern.name}**")
            st.write(f"  Type: {pattern.pattern_type.value}")
        
        if st.button("🔄 Reset"):
            st.session_state.gaps_data = None
            st.session_state.results = None
            st.rerun()
    
    # Interface principale
    tab1, tab2, tab3 = st.tabs(["🔍 Gaps d'Exemple", "🎨 Application", "📊 Résultats"])
    
    with tab1:
        st.header("🔍 Génération de Gaps d'Exemple")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Configuration")
            num_gaps = st.slider("Nombre de gaps", 3, 10, 6)
            
            if st.button("🎯 Générer Gaps Réalistes", type="primary"):
                # Créer gaps d'exemple
                example_gaps = [
                    Gap(
                        source_name="PremiumCustomer",
                        source_type="Class",
                        target_name=None,
                        target_type=None,
                        similarity=0.0,
                        severity=0.8,
                        gap_type="structural",
                        properties_lost={"discountRate": 0.15, "membershipLevel": "GOLD"},
                        constraints_lost=["OCL: inv discountRate >= 0", "OCL: inv membershipLevel <> null"]
                    ),
                    Gap(
                        source_name="validateEmail",
                        source_type="Operation",
                        target_name=None,
                        target_type=None,
                        similarity=0.0,
                        severity=0.7,
                        gap_type="behavioral",
                        properties_lost={"return_type": "boolean", "visibility": "public"},
                        constraints_lost=["OCL: pre: email <> null", "OCL: post: result = email.matches('.*@.*')"]
                    ),
                    Gap(
                        source_name="phoneNumber",
                        source_type="Attribute",
                        target_name=None,
                        target_type=None,
                        similarity=0.0,
                        severity=0.5,
                        gap_type="metadata",
                        properties_lost={"data_type": "String", "validation": "phone_format"},
                        constraints_lost=["validation: phone format required"]
                    ),
                    Gap(
                        source_name="OrderStatus",
                        source_type="Enumeration",
                        target_name=None,
                        target_type=None,
                        similarity=0.0,
                        severity=0.9,
                        gap_type="structural",
                        properties_lost={"literals": ["PENDING", "CONFIRMED", "DELIVERED"]},
                        constraints_lost=["constraint: default = PENDING"]
                    )
                ]
                
                # Ajouter gaps supplémentaires si demandé
                if num_gaps > 4:
                    for i in range(4, num_gaps):
                        example_gaps.append(Gap(
                            source_name=f"Element_{i}",
                            source_type="Attribute",
                            target_name=None,
                            target_type=None,
                            similarity=0.0,
                            severity=0.6,
                            gap_type="metadata",
                            properties_lost={f"prop_{i}": f"value_{i}"},
                            constraints_lost=[f"constraint_{i}"]
                        ))
                
                st.session_state.gaps_data = example_gaps[:num_gaps]
                st.success(f"✅ {len(st.session_state.gaps_data)} gaps générés!")
        
        with col2:
            if st.session_state.gaps_data:
                st.subheader("📋 Gaps Générés")
                for i, gap in enumerate(st.session_state.gaps_data, 1):
                    st.write(f"**{i}. {gap.source_name}** ({gap.source_type})")
                    st.write(f"Sévérité: {gap.severity:.2f}, Type: {gap.gap_type}")
                    if gap.properties_lost:
                        st.write(f"Propriétés: {len(gap.properties_lost)}")
                    if gap.constraints_lost:
                        st.write(f"Contraintes: {len(gap.constraints_lost)}")
                    st.write("---")
    
    with tab2:
        st.header("🎨 Application des Patterns")
        
        if st.session_state.gaps_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚙️ Configuration")
                ba_initial = st.slider("Score BA initial", 0.0, 1.0, 0.45, 0.01)
                
                base_model = st.text_area(
                    "Modèle de base:",
                    height=200,
                    value=f"""<?xml version="1.0" encoding="UTF-8"?>
<ecore:EPackage name="example">

<eClass name="Customer">
    <eAttribute name="name" eType="String"/>
    <eAttribute name="email" eType="String"/>
</eClass>

</ecore:EPackage>""" if target_metamodel == "Ecore" else """package com.example;

public class Customer {
    private String name;
    private String email;
}"""
                )
            
            with col2:
                st.subheader("📊 Aperçu")
                st.write(f"**Gaps à traiter:** {len(st.session_state.gaps_data)}")
                st.write(f"**Métamodèle cible:** {target_metamodel}")
                st.write(f"**Patterns disponibles:** {len(engine.patterns)}")
            
            if st.button("🚀 APPLIQUER LES PATTERNS", type="primary", use_container_width=True):
                with st.spinner("🎨 Application en cours..."):
                    results = engine.apply_patterns(
                        st.session_state.gaps_data,
                        base_model,
                        target_metamodel
                    )
                    results['ba_initial'] = ba_initial
                    st.session_state.results = results
                
                st.success("✅ Patterns appliqués!")
                
                # Résumé immédiat
                if st.session_state.results:
                    r = st.session_state.results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Applications", r['total_applications'])
                    with col2:
                        st.metric("Succès", r['successful_applications'])
                    with col3:
                        st.metric("Amélioration", f"+{r['total_improvement']*100:.1f}%")
                    with col4:
                        st.metric("Taux Succès", f"{r['success_rate']:.0%}")
        else:
            st.warning("⚠️ Générez d'abord des gaps dans l'onglet précédent")
    
    with tab3:
        st.header("📊 Résultats et Code Généré")
        
        if st.session_state.results:
            results = st.session_state.results
            
            # Graphique des améliorations
            fig = create_improvement_chart(results['applications'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Applications détaillées
            st.subheader("🔍 Applications Détaillées")
            
            for i, app in enumerate(results['applications'], 1):
                css_class = "pattern-success" if app.success else "pattern-failed"
                
                with st.container():
                    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{i}. {app.pattern_name}**")
                        st.write(f"Élément: {app.source_element}")
                        st.write(f"*{app.explanation}*")
                    
                    with col2:
                        if app.success:
                            st.markdown(f'<span class="improvement-badge">+{app.improvement_score:.1%}</span>', 
                                      unsafe_allow_html=True)
                            st.write(f"Complexité: +{app.complexity_added:.3f}")
                        else:
                            st.write("❌ Échec")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Code final généré
            st.subheader("💾 Code Final Généré")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown('<div class="code-block">', unsafe_allow_html=True)
                st.code(results['final_model'], language='xml' if target_metamodel == 'Ecore' else 'java')
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                final_code = results['final_model']
                lines = len(final_code.split('\n'))
                chars = len(final_code)
                
                st.metric("Lignes", lines)
                st.metric("Caractères", chars)
                
                # Téléchargement
                file_ext = '.ecore' if target_metamodel == 'Ecore' else '.java'
                st.download_button(
                    "💾 Télécharger",
                    data=final_code,
                    file_name=f"model_corrected{file_ext}",
                    mime="text/plain"
                )
            
            # Métriques finales
            st.subheader("📈 Impact Global")
            
            ba_initial = results['ba_initial']
            ba_final = ba_initial + results['total_improvement']
            improvement_pct = (results['total_improvement'] / ba_initial) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Score BA Initial", f"{ba_initial:.3f}")
            with col2:
                st.metric("Score BA Final", f"{min(ba_final, 1.0):.3f}", 
                         delta=f"+{results['total_improvement']:.3f}")
            with col3:
                st.metric("Gain Relatif", f"+{improvement_pct:.1f}%")
        
        else:
            st.info("🔄 Appliquez d'abord des patterns pour voir les résultats")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <h4>🎨 Framework de Patterns Concrets v1.0</h4>
        <p><strong>3 patterns</strong> • <strong>Correction automatique</strong> • <strong>Code fonctionnel</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()