#!/usr/bin/env python3
"""
Clean Real Evaluation - No Streamlit-PyTorch Conflicts
Simplified version that works reliably
"""

import os
import sys
import warnings
import json
import time
from pathlib import Path

# Clean environment setup
warnings.filterwarnings('ignore')
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'TOKENIZERS_PARALLELISM': 'false',
    'PYTHONWARNINGS': 'ignore'
})

# Import order matters - Streamlit first
import streamlit as st

# Configure Streamlit immediately
st.set_page_config(
    page_title="🔬 Real Semantic Evaluation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe ML imports
def safe_import_ml():
    """Safely import ML libraries"""
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertModel
        from sklearn.metrics.pairwise import cosine_similarity
        return True, (torch, DistilBertTokenizer, DistilBertModel, cosine_similarity)
    except Exception:
        return False, None

# Test ML availability
ML_AVAILABLE, ML_MODULES = safe_import_ml()

def create_clean_interface():
    """Clean interface without conflicts"""
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 2.5rem; margin: 0;'>🔬 Real Semantic Evaluation</h1>
        <p style='color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>Conflict-Free DistilBERT + Pattern Application</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ML_AVAILABLE:
            st.success("✅ ML Libraries Ready")
        else:
            st.error("❌ ML Libraries Missing")
    
    with col2:
        modelset_exists = Path("modelset").exists()
        if modelset_exists:
            file_count = len(list(Path("modelset").rglob("*.ecore"))) + \
                        len(list(Path("modelset").rglob("*.uml"))) + \
                        len(list(Path("modelset").rglob("*.java")))
            if file_count > 0:
                st.success(f"✅ ModelSet: {file_count} files")
            else:
                st.warning("⚠️ ModelSet: No model files")
        else:
            st.warning("⚠️ ModelSet: Not found")
    
    with col3:
        if ML_AVAILABLE:
            torch = ML_MODULES[0]
            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.info(f"🖥️ Device: {device}")
        else:
            st.info("🖥️ Device: N/A")
    
    # Main interface
    if not ML_AVAILABLE:
        st.error("""
        ❌ **Cannot Run Real Evaluation**
        
        ML libraries are not properly installed or configured.
        
        **To fix this:**
        ```bash
        pip install torch transformers scikit-learn
        ```
        
        **Alternative:** Use the simulation mode in `modelset_evaluator.py`
        """)
        return
    
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        modelset_path = st.text_input("ModelSet Directory", value="modelset")
        max_models = st.slider("Models to Evaluate", 1, 10, 3)
        
    with col2:
        use_gpu = st.checkbox("Use GPU if Available", value=True)
        batch_size = st.slider("Batch Size", 2, 16, 4)
    
    # Evaluation button
    if st.button("🚀 START REAL EVALUATION", type="primary"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import and run evaluation
            status_text.text("🔄 Initializing components...")
            
            # Import the evaluator
            from full_real_implementation import FullRealEvaluator
            
            progress_bar.progress(0.1)
            
            # Create evaluator
            evaluator = FullRealEvaluator(modelset_path)
            
            status_text.text("🧠 Initializing DistilBERT...")
            if not evaluator.initialize():
                st.error("❌ Failed to initialize DistilBERT")
                return
            
            progress_bar.progress(0.3)
            
            # Run evaluation
            status_text.text("🧪 Running evaluation...")
            
            def progress_callback(msg):
                status_text.text(f"🔄 {msg}")
            
            results = evaluator.run_full_evaluation(
                max_models=max_models,
                progress_callback=progress_callback
            )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Evaluation completed!")
            
            # Display results
            if results:
                st.success(f"🎉 Successfully evaluated {len(results)} models!")
                
                # Summary metrics
                avg_improvement = sum(r.improvement_percentage for r in results) / len(results)
                avg_ba_initial = sum(r.ba_score_initial for r in results) / len(results)
                avg_ba_final = sum(r.ba_score_final for r in results) / len(results)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Improvement", f"{avg_improvement:.1f}%")
                with col2:
                    st.metric("Avg BA Initial", f"{avg_ba_initial:.3f}")
                with col3:
                    st.metric("Avg BA Final", f"{avg_ba_final:.3f}")
                with col4:
                    st.metric("Models Processed", len(results))
                
                # Results table
                st.subheader("📊 Detailed Results")
                
                results_data = []
                for r in results:
                    results_data.append({
                        'Model': r.model_id,
                        'Type': r.transformation_type,
                        'BA Initial': f"{r.ba_score_initial:.3f}",
                        'BA Final': f"{r.ba_score_final:.3f}",
                        'Improvement': f"{r.improvement_percentage:.1f}%",
                        'Gaps': r.gaps_detected,
                        'Patterns': len(r.patterns_applied),
                        'Time (s)': f"{r.processing_time:.1f}"
                    })
                
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Export results
                results_json = json.dumps([{
                    'model_id': r.model_id,
                    'transformation_type': r.transformation_type,
                    'ba_score_initial': r.ba_score_initial,
                    'ba_score_final': r.ba_score_final,
                    'improvement_percentage': r.improvement_percentage,
                    'gaps_detected': r.gaps_detected,
                    'patterns_applied': r.patterns_applied,
                    'real_evaluation': True
                } for r in results], indent=2)
                
                st.download_button(
                    "💾 Download Results",
                    data=results_json,
                    file_name="real_evaluation_results.json",
                    mime="application/json"
                )
                
                # Success summary
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 10px; color: white; margin: 1rem 0;'>
                    <h3>🎉 Real Evaluation Successful!</h3>
                    <p><strong>Average Improvement:</strong> {avg_improvement:.1f}% BA score increase</p>
                    <p><strong>Models Evaluated:</strong> {len(results)} with real DistilBERT embeddings</p>
                    <p><strong>Patterns Applied:</strong> Authentic pattern corrections applied</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error("❌ No results obtained from evaluation")
                
        except Exception as e:
            st.error(f"❌ Evaluation failed: {str(e)}")
            
            # Show limited error info
            st.code(f"Error type: {type(e).__name__}")
            
            st.info("""
            💡 **Troubleshooting:**
            1. Ensure ModelSet directory exists and contains model files
            2. Check that ML libraries are properly installed
            3. Try reducing the number of models
            4. Use simulation mode if real evaluation continues to fail
            """)

if __name__ == "__main__":
    create_clean_interface()