#!/usr/bin/env python3
"""
Extraction de Token Pairs Améliorée
Correction du problème d'extraction pour obtenir plus de token pairs source
"""

import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import json
import time

# Configuration pour l'intégration legacy
LEGACY_REPO_PATH = "trans-neur-emb-sem-pre-meta-trans"
if os.path.exists(LEGACY_REPO_PATH):
    sys.path.insert(0, LEGACY_REPO_PATH)

try:
    from token_pair import TokenPair as LegacyTokenPair
    from token_pair import TokenPairExtractor, TokenPairSimilarityCalculator
    from embedding import EmbeddingModel
    from semantic_preservation import SemanticPreservationFramework as LegacyFramework
    LEGACY_AVAILABLE = True
    print("✅ Modules legacy importés avec succès")
except ImportError as e:
    print(f"⚠️  Import legacy échoué: {e}")
    LEGACY_AVAILABLE = False

@dataclass
class EnrichedTokenPair:
    """Version corrigée avec meilleure extraction"""
    element_name: str
    element_type: str
    metamodel_info: Dict = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    context_hierarchy: List[str] = field(default_factory=list)
    semantic_properties: Dict = field(default_factory=dict)
    context_importance: float = 0.0
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.context_importance == 0.0:
            self.context_importance = self._compute_context_importance()
    
    def _compute_context_importance(self) -> float:
        importance = 0.5
        if self.constraints:
            importance += 0.2 * min(len(self.constraints) / 3, 1.0)
        if self.relationships:
            importance += 0.2 * min(len(self.relationships) / 5, 1.0)
        if self.context_hierarchy:
            importance += 0.1 * min(len(self.context_hierarchy) / 4, 1.0)
        if self.semantic_properties:
            importance += 0.1 * min(len(self.semantic_properties) / 3, 1.0)
        return min(importance, 1.0)
    
    def serialize_for_embedding(self) -> str:
        base = f"{self.element_type}: {self.element_name}"
        enrichments = []
        
        if self.semantic_properties:
            props = ", ".join([f"{k}={v}" for k,v in self.semantic_properties.items()])
            enrichments.append(f"properties: {props}")
        
        if self.constraints:
            constraints = "; ".join(self.constraints)
            enrichments.append(f"constraints: {constraints}")
        
        if self.context_hierarchy:
            hierarchy = " -> ".join(self.context_hierarchy)
            enrichments.append(f"context: {hierarchy}")
        
        if self.relationships:
            rel_types = [rel.get('type', 'unknown') for rel in self.relationships]
            relationships = ", ".join(set(rel_types))
            enrichments.append(f"relationships: {relationships}")
        
        if enrichments:
            return f"{base} [{'; '.join(enrichments)}]"
        return base
    
    def to_dict(self) -> Dict:
        return {
            'element_name': self.element_name,
            'element_type': self.element_type,
            'metamodel_info': self.metamodel_info,
            'constraints': self.constraints,
            'relationships': self.relationships,
            'context_hierarchy': self.context_hierarchy,
            'semantic_properties': self.semantic_properties,
            'context_importance': self.context_importance,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }

class ImprovedTokenPairExtractor:
    """Extracteur amélioré avec patterns plus robustes"""
    
    def __init__(self):
        # Patterns améliorés pour différents métamodèles
        self.patterns = {
            'UML': {
                'class': [
                    r'class\s+(\w+)',
                    r'public\s+class\s+(\w+)',
                    r'abstract\s+class\s+(\w+)',
                    r'final\s+class\s+(\w+)'
                ],
                'interface': [
                    r'interface\s+(\w+)',
                    r'public\s+interface\s+(\w+)'
                ],
                'attribute': [
                    r'(\w+)\s+(\w+)\s*;',  # type name;
                    r'private\s+(\w+)\s+(\w+)',
                    r'public\s+(\w+)\s+(\w+)',
                    r'protected\s+(\w+)\s+(\w+)',
                    r'(\w+)\s+(\w+)\s*=',  # with initialization
                ],
                'operation': [
                    r'(\w+)\s+(\w+)\s*\(',  # return_type method_name(
                    r'public\s+(\w+)\s+(\w+)\s*\(',
                    r'private\s+(\w+)\s+(\w+)\s*\(',
                    r'void\s+(\w+)\s*\(',  # void method_name(
                    r'(\w+)\[\]\s+(\w+)\s*\('  # array return type
                ]
            },
            'Ecore': {
                'eclass': [
                    r'<eClass\s+name="(\w+)"',
                    r'eClass\s+name="(\w+)"',
                    r'<ecore:EClass\s+name="(\w+)"'
                ],
                'eattribute': [
                    r'<eAttribute\s+name="(\w+)"',
                    r'eAttribute\s+name="(\w+)"',
                    r'<ecore:EAttribute\s+name="(\w+)"'
                ],
                'ereference': [
                    r'<eReference\s+name="(\w+)"',
                    r'eReference\s+name="(\w+)"',
                    r'<ecore:EReference\s+name="(\w+)"'
                ],
                'eoperation': [
                    r'<eOperation\s+name="(\w+)"',
                    r'eOperation\s+name="(\w+)"'
                ]
            },
            'Java': {
                'class': [
                    r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)',
                    r'class\s+(\w+)\s+extends',
                    r'class\s+(\w+)\s+implements'
                ],
                'interface': [
                    r'(?:public\s+)?interface\s+(\w+)'
                ],
                'field': [
                    r'(?:private|public|protected)\s+(?:static\s+)?(?:final\s+)?(\w+)\s+(\w+)',
                    r'(\w+)\s+(\w+)\s*=.*?;'
                ],
                'method': [
                    r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(\w+)\s+(\w+)\s*\(',
                    r'(?:public|private|protected)\s+(\w+)\s*\('
                ]
            }
        }
    
    def extract_from_text(self, text: str, metamodel: str = "UML") -> List[EnrichedTokenPair]:
        """Extraction améliorée avec patterns multiples"""
        print(f"🔍 Extraction token pairs pour {metamodel}")
        print(f"   Texte à analyser: {len(text)} caractères")
        
        # Nettoyer le texte
        cleaned_text = self._clean_text(text)
        print(f"   Texte nettoyé: {len(cleaned_text)} caractères")
        
        extracted_pairs = []
        metamodel_patterns = self.patterns.get(metamodel, self.patterns['UML'])
        
        for element_type, patterns in metamodel_patterns.items():
            pairs_for_type = self._extract_by_patterns(cleaned_text, patterns, element_type, metamodel)
            extracted_pairs.extend(pairs_for_type)
            print(f"   {element_type}: {len(pairs_for_type)} éléments")
        
        # Enrichir les token pairs
        enriched_pairs = []
        for tp in extracted_pairs:
            enriched_tp = self._enrich_token_pair(tp, cleaned_text)
            enriched_pairs.append(enriched_tp)
        
        print(f"✅ Total: {len(enriched_pairs)} token pairs extraits")
        return enriched_pairs
    
    def _clean_text(self, text: str) -> str:
        """Nettoie le texte pour améliorer l'extraction"""
        # Supprimer les commentaires
        text = re.sub(r'//.*?\n', '\n', text)  # Commentaires //
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # Commentaires /* */
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _extract_by_patterns(self, text: str, patterns: List[str], element_type: str, metamodel: str) -> List[EnrichedTokenPair]:
        """Extrait avec plusieurs patterns pour robustesse"""
        found_elements = set()  # Éviter les doublons
        token_pairs = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                if isinstance(match, tuple):
                    # Pattern avec plusieurs groupes
                    if len(match) >= 2:
                        # Pour les attributs/méthodes: (type, name) ou (visibility, type, name)
                        element_name = match[-1]  # Dernier élément = nom
                        type_info = match[0] if len(match) >= 2 else "unknown"
                    else:
                        element_name = match[0]
                        type_info = element_type
                else:
                    # Pattern avec un seul groupe
                    element_name = match
                    type_info = element_type
                
                # Éviter les doublons et les noms vides
                if element_name and element_name not in found_elements and len(element_name) > 1:
                    found_elements.add(element_name)
                    
                    tp = EnrichedTokenPair(
                        element_name=element_name,
                        element_type=element_type.title(),  # Capitaliser
                        metamodel_info={
                            'type': metamodel,
                            'original_type': type_info if isinstance(type_info, str) else element_type
                        }
                    )
                    token_pairs.append(tp)
        
        return token_pairs
    
    def _enrich_token_pair(self, tp: EnrichedTokenPair, full_text: str) -> EnrichedTokenPair:
        """Enrichit un token pair avec des informations contextuelles"""
        # Rechercher le contexte autour de l'élément
        element_context = self._find_element_context(tp.element_name, full_text)
        
        # Inférer des propriétés sémantiques
        tp.semantic_properties = self._infer_semantic_properties(tp, element_context)
        
        # Inférer des contraintes
        tp.constraints = self._infer_constraints(tp, element_context)
        
        # Inférer des relations
        tp.relationships = self._infer_relationships(tp, element_context)
        
        # Inférer la hiérarchie contextuelle
        tp.context_hierarchy = self._infer_context_hierarchy(tp, full_text)
        
        # Recalculer l'importance
        tp.context_importance = tp._compute_context_importance()
        
        return tp
    
    def _find_element_context(self, element_name: str, text: str, context_size: int = 200) -> str:
        """Trouve le contexte autour d'un élément"""
        # Chercher l'élément dans le texte
        pattern = rf'\b{re.escape(element_name)}\b'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            start = max(0, match.start() - context_size)
            end = min(len(text), match.end() + context_size)
            return text[start:end]
        
        return ""
    
    def _infer_semantic_properties(self, tp: EnrichedTokenPair, context: str) -> Dict[str, Any]:
        """Infère des propriétés sémantiques"""
        properties = {}
        
        context_lower = context.lower()
        element_name_lower = tp.element_name.lower()
        
        # Propriétés basées sur les mots-clés
        if 'abstract' in context_lower:
            properties['abstract'] = True
        
        if 'static' in context_lower:
            properties['static'] = True
        
        if 'final' in context_lower:
            properties['final'] = True
        
        if 'private' in context_lower:
            properties['visibility'] = 'private'
        elif 'protected' in context_lower:
            properties['visibility'] = 'protected'
        elif 'public' in context_lower:
            properties['visibility'] = 'public'
        
        # Propriétés basées sur le nom
        if 'id' in element_name_lower or element_name_lower.endswith('id'):
            properties['identifier'] = True
        
        if element_name_lower.startswith('is') or element_name_lower.startswith('has'):
            properties['boolean_property'] = True
        
        if element_name_lower.startswith('get') or element_name_lower.startswith('set'):
            properties['accessor'] = True
        
        # Propriétés dérivées
        if '/derived' in context or 'derived' in context_lower:
            properties['derived'] = True
        
        return properties
    
    def _infer_constraints(self, tp: EnrichedTokenPair, context: str) -> List[str]:
        """Infère des contraintes"""
        constraints = []
        
        # Contraintes OCL
        ocl_patterns = [
            r'inv\s+\w*\s*:([^}]+)',
            r'pre\s+\w*\s*:([^}]+)',
            r'post\s+\w*\s*:([^}]+)'
        ]
        
        for pattern in ocl_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            for match in matches:
                constraint = match.strip()
                if constraint and len(constraint) < 100:  # Éviter les contraintes trop longues
                    constraints.append(f"OCL: {constraint}")
        
        # Contraintes de validation simples
        if tp.element_type.lower() in ['attribute', 'eattribute'] and tp.element_name:
            name_lower = tp.element_name.lower()
            if 'email' in name_lower:
                constraints.append("validation: email format required")
            if 'age' in name_lower:
                constraints.append("validation: age >= 0")
            if 'password' in name_lower:
                constraints.append("validation: minimum length required")
        
        return constraints[:3]  # Limiter à 3 contraintes
    
    def _infer_relationships(self, tp: EnrichedTokenPair, context: str) -> List[Dict[str, Any]]:
        """Infère des relations"""
        relationships = []
        
        # Mots-clés de relations
        relation_patterns = {
            'inheritance': [r'extends\s+(\w+)', r'inherits\s+from\s+(\w+)'],
            'implementation': [r'implements\s+(\w+)'],
            'association': [r'has\s+(\w+)', r'contains\s+(\w+)'],
            'aggregation': [r'aggregates\s+(\w+)'],
            'composition': [r'composes\s+(\w+)', r'composed\s+of\s+(\w+)'],
            'dependency': [r'uses\s+(\w+)', r'depends\s+on\s+(\w+)']
        }
        
        for relation_type, patterns in relation_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                for match in matches:
                    target = match if isinstance(match, str) else match[0]
                    if target and target != tp.element_name:
                        relationships.append({
                            'type': relation_type,
                            'target': target,
                            'inferred': True
                        })
        
        return relationships[:5]  # Limiter à 5 relations
    
    def _infer_context_hierarchy(self, tp: EnrichedTokenPair, full_text: str) -> List[str]:
        """Infère la hiérarchie contextuelle"""
        hierarchy = []
        
        # Rechercher des namespaces ou packages
        namespace_patterns = [
            r'package\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            r'namespace\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            r'([a-zA-Z_][a-zA-Z0-9_.]*)::'
        ]
        
        for pattern in namespace_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                namespace = match if isinstance(match, str) else match[0]
                if '.' in namespace:
                    hierarchy.extend(namespace.split('.'))
                else:
                    hierarchy.append(namespace)
        
        # Déduplication et limitation
        hierarchy = list(dict.fromkeys(hierarchy))  # Préserver l'ordre et supprimer doublons
        return hierarchy[:4]

def test_improved_extraction():
    """Test de l'extraction améliorée"""
    print("🧪 Test de l'Extraction Améliorée")
    print("="*50)
    
    # Créer l'extracteur
    extractor = ImprovedTokenPairExtractor()
    
    # Modèles de test plus réalistes
    uml_model = """
    package com.example.model;
    
    public abstract class Customer {
        private String name;
        private int age;
        private Email email;
        
        // OCL: inv: age >= 0
        // OCL: inv: email.isValid()
        
        public void updateProfile();
        public Order[] getOrders();
        protected boolean isActive();
    }
    
    public class Order extends Transaction {
        private Date orderDate;
        private double totalAmount;
        private OrderStatus status;
        
        public Order(Customer customer) {
            // constructor
        }
        
        public void calculateTotal();
        public boolean isCompleted();
    }
    
    public interface PaymentProcessor {
        boolean processPayment(double amount);
        void refund(String transactionId);
    }
    """
    
    ecore_model = """
    <?xml version="1.0" encoding="UTF-8"?>
    <ecore:EPackage xmi:version="2.0" 
        xmlns:xmi="http://www.omg.org/XMI" 
        xmlns:ecore="http://www.eclipse.org/emf/2002/Ecore"
        name="customermodel">
        
    <eClass name="Customer" abstract="true">
        <eAttribute name="name" eType="ecore:EDataType String"/>
        <eAttribute name="age" eType="ecore:EDataType int"/>
        <eReference name="orders" eType="#//Order" upperBound="-1"/>
        <eOperation name="updateProfile"/>
        <eOperation name="getOrders" eType="#//Order" upperBound="-1"/>
    </eClass>
    
    <eClass name="Order">
        <eAttribute name="orderDate" eType="ecore:EDataType Date"/>
        <eAttribute name="totalAmount" eType="ecore:EDataType double"/>
        <eAttribute name="status" eType="#//OrderStatus"/>
        <eReference name="customer" eType="#//Customer"/>
        <eOperation name="calculateTotal"/>
    </eClass>
    
    <eClass name="PaymentProcessor" interface="true">
        <eOperation name="processPayment" eType="ecore:EDataType boolean">
            <eParameter name="amount" eType="ecore:EDataType double"/>
        </eOperation>
    </eClass>
    
    </ecore:EPackage>
    """
    
    print("📋 Test UML:")
    print(f"   Taille: {len(uml_model)} caractères")
    uml_pairs = extractor.extract_from_text(uml_model, "UML")
    
    print(f"\n📊 Résultats UML:")
    for tp in uml_pairs:
        print(f"   • {tp.element_type}: {tp.element_name}")
        if tp.semantic_properties:
            props = ", ".join([f"{k}={v}" for k,v in tp.semantic_properties.items()])
            print(f"     Properties: {props}")
        if tp.constraints:
            print(f"     Constraints: {len(tp.constraints)}")
        if tp.relationships:
            print(f"     Relationships: {len(tp.relationships)}")
    
    print(f"\n📋 Test Ecore:")
    print(f"   Taille: {len(ecore_model)} caractères")
    ecore_pairs = extractor.extract_from_text(ecore_model, "Ecore")
    
    print(f"\n📊 Résultats Ecore:")
    for tp in ecore_pairs:
        print(f"   • {tp.element_type}: {tp.element_name}")
        if tp.semantic_properties:
            props = ", ".join([f"{k}={v}" for k,v in tp.semantic_properties.items()])
            print(f"     Properties: {props}")
    
    print(f"\n✅ Comparaison:")
    print(f"   UML pairs: {len(uml_pairs)}")
    print(f"   Ecore pairs: {len(ecore_pairs)}")
    
    # Test de sérialisation pour embeddings
    print(f"\n🧠 Test sérialisation pour embeddings:")
    if uml_pairs:
        sample_tp = uml_pairs[0]
        serialized = sample_tp.serialize_for_embedding()
        print(f"   Exemple: {serialized}")
    
    return uml_pairs, ecore_pairs

def demo_with_improved_extraction():
    """Démonstration avec extraction améliorée"""
    print("🚀 Démonstration avec Extraction Améliorée")
    print("="*60)
    
    # Tester l'extraction d'abord
    uml_pairs, ecore_pairs = test_improved_extraction()
    
    if not uml_pairs:
        print("❌ Aucun token pair UML extrait - problème d'extraction")
        return None
    
    print(f"\n🔍 Analyse des gaps...")
    # Simulation simple de détection de gaps
    gaps_found = []
    
    # Créer des maps pour recherche rapide
    uml_names = {tp.element_name.lower(): tp for tp in uml_pairs}
    ecore_names = {tp.element_name.lower(): tp for tp in ecore_pairs}
    
    # Chercher les éléments UML sans correspondance exacte en Ecore
    for uml_tp in uml_pairs:
        uml_name = uml_tp.element_name.lower()
        if uml_name not in ecore_names:
            # Chercher correspondance approximative
            best_match = None
            best_score = 0.0
            
            for ecore_name, ecore_tp in ecore_names.items():
                # Similarité simple basée sur les noms
                if uml_name in ecore_name or ecore_name in uml_name:
                    score = 0.6
                elif any(word in ecore_name for word in uml_name.split()):
                    score = 0.4
                else:
                    score = 0.1
                
                if score > best_score:
                    best_score = score
                    best_match = ecore_tp
            
            if best_score < 0.5:  # Seuil de gap
                gap_info = {
                    'source': uml_tp.element_name,
                    'source_type': uml_tp.element_type,
                    'best_match': best_match.element_name if best_match else None,
                    'similarity': best_score,
                    'gap_severity': 1.0 - best_score
                }
                gaps_found.append(gap_info)
    
    print(f"\n📊 Résultats de l'analyse:")
    print(f"   Token pairs UML: {len(uml_pairs)}")
    print(f"   Token pairs Ecore: {len(ecore_pairs)}")
    print(f"   Gaps détectés: {len(gaps_found)}")
    
    if gaps_found:
        print(f"\n🔍 Gaps principaux:")
        for gap in gaps_found[:5]:
            print(f"   • {gap['source']} ({gap['source_type']}) - similarité: {gap['similarity']:.2f}")
            if gap['best_match']:
                print(f"     Meilleure correspondance: {gap['best_match']}")
    
    # Calculer métriques simulées
    coverage = 1.0 - (len(gaps_found) / max(len(uml_pairs), 1))
    avg_similarity = sum(gap['similarity'] for gap in gaps_found) / max(len(gaps_found), 1)
    
    print(f"\n📈 Métriques calculées:")
    print(f"   Couverture: {coverage:.2%}")
    print(f"   Similarité moyenne gaps: {avg_similarity:.3f}")
    print(f"   Score de préservation estimé: {(coverage + avg_similarity) / 2:.3f}")
    
    return {
        'uml_pairs': len(uml_pairs),
        'ecore_pairs': len(ecore_pairs),
        'gaps_found': len(gaps_found),
        'coverage': coverage,
        'preservation_score': (coverage + avg_similarity) / 2
    }

if __name__ == "__main__":
    # Exécuter la démonstration améliorée
    result = demo_with_improved_extraction()
    
    if result:
        print(f"\n💾 Résultats:")
        print(f"   ✅ Extraction UML réussie: {result['uml_pairs']} éléments")
        print(f"   ✅ Extraction Ecore réussie: {result['ecore_pairs']} éléments") 
        print(f"   📊 Score de préservation: {result['preservation_score']:.1%}")
        
        # Sauvegarder les résultats
        with open("improved_extraction_results.json", 'w') as f:
            json.dump(result, f, indent=2)
        print(f"   💾 Résultats sauvegardés dans improved_extraction_results.json")
    else:
        print("❌ Problème avec l'extraction - vérifier les patterns")