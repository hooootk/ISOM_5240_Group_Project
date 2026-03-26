"""
Streamlit Application: AI Content Generation and Safety Detection System
Title: Real-time Safe Checks for AI Agents
"""

import streamlit as st
import torch
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

# Suppress warnings
warnings.filterwarnings('ignore')


# ============================================
# Configuration Constants
# ============================================
MODEL_A_NAME = "facebook/opt-350m"
MODEL_B_PATH = "./distilbert-base-uncased-beavertails-final"


# ============================================
# Model Loading Functions
# ============================================
@st.cache_resource
def load_model_a():
    """Load text generation model (OPT-350M)"""
    try:
        generator = pipeline(
            "text-generation",
            model=MODEL_A_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return generator
    except Exception as e:
        st.error(f"Failed to load Model OPT-350M: {e}")
        return None


@st.cache_resource
def load_model_b():
    """Load safety classification model (DistilBERT)"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_B_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_B_PATH)
        model = model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Failed to load Model Fine-tuned DistilBERT: {e}")
        return None, None, None


# ============================================
# Content Generation Function
# ============================================
def generate_content(generator, prompt):
    """
    Generate content using OPT-350M with specified parameters
    
    Args:
        generator: The text generation pipeline
        prompt: User input prompt
    
    Returns:
        tuple: (generated_text, error_message)
    """
    try:
        result = generator(
            prompt,
            max_new_tokens=150,
            return_full_text=False,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1
        )
        
        generated_text = result[0]['generated_text'].strip()
        return generated_text, None
    except Exception as e:
        return None, str(e)


# ============================================
# Safety Detection Function
# ============================================
def check_safety(tokenizer, model, device, prompt, response):
    """
    Check if the generated content is safe using DistilBERT
    
    Args:
        tokenizer: The tokenizer for the safety model
        model: The safety classification model
        device: Device to run inference on (cuda/cpu)
        prompt: Original user prompt
        response: Generated content to check
    
    Returns:
        tuple: (is_safe, confidence, error_message)
    """
    try:
        text = f"{prompt} [SEP] {response}"
        
        inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()
        
        # pred=0 means safe, pred=1 means unsafe
        is_safe = (pred == 0)
        return is_safe, confidence, None
        
    except Exception as e:
        return False, 0.0, str(e)


# ============================================
# Session State Initialization
# ============================================
def init_session_state():
    """Initialize session state variables"""
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = None
    if 'safety_result' not in st.session_state:
        st.session_state.safety_result = None
    if 'confidence_score' not in st.session_state:
        st.session_state.confidence_score = None
    if 'show_blocked' not in st.session_state:
        st.session_state.show_blocked = False
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = None


# ============================================
# Clear Output Function
# ============================================
def clear_output():
    """Clear all output content from session state"""
    st.session_state.generated_content = None
    st.session_state.safety_result = None
    st.session_state.confidence_score = None
    st.session_state.show_blocked = False


# ============================================
# Show Blocked Content Function
# ============================================
def show_blocked_content():
    """Toggle visibility of blocked content"""
    st.session_state.show_blocked = True


# ============================================
# UI Rendering Function
# ============================================
def render_ui():
    """Render Streamlit interface"""
    # Page configuration
    st.set_page_config(
        page_title="ABCDEFG",
        page_icon="🛡️"
    )
    
    # Initialize session state
    init_session_state()
    
    # Title
    st.title("Real-time Safe Checks for AI Agents")
    
    # Check if model exists
    if not os.path.exists(MODEL_B_PATH):
        st.error(f"Model not found: {MODEL_B_PATH}")
        st.stop()
    
    # Load models
    with st.spinner("Loading models..."):
        generator = load_model_a()
        tokenizer, model_b, device = load_model_b()
    
    if generator is None or tokenizer is None or model_b is None:
        st.stop()
    
    # Input area
    prompt = st.text_input(
        "Enter your prompt:",
        placeholder="e.g., Write an article about artificial intelligence",
        key="prompt_input"
    )
    
    # Generate button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Generate", type="primary"):
            if not prompt:
                st.warning("Please enter a prompt")
            else:
                # Clear previous output before new generation
                clear_output()
                st.session_state.current_prompt = prompt
                
                # Generate content
                with st.spinner("Generating content..."):
                    generated_content, gen_error = generate_content(generator, prompt)
                
                if gen_error:
                    st.error(gen_error)
                else:
                    # Safety detection
                    with st.spinner("Checking safety..."):
                        is_safe, confidence, safety_error = check_safety(
                            tokenizer, model_b, device, prompt, generated_content
                        )
                    
                    if safety_error:
                        st.error(safety_error)
                    else:
                        # Store results in session state
                        st.session_state.generated_content = generated_content
                        st.session_state.safety_result = is_safe
                        st.session_state.confidence_score = confidence
                        
                        # Force a rerun to display results
                        st.rerun()
    
    # Display results if they exist
    if st.session_state.generated_content is not None:
        st.markdown("---")
        
        # Display safety detection result
        st.subheader("Safety Detection")
        
        if st.session_state.safety_result:
            # Safe content - display directly
            st.success(f"✅ **Safe**\n\nConfidence: {st.session_state.confidence_score:.4f}")
            st.subheader("Generated Content")
            st.write(st.session_state.generated_content)
        else:
            # Unsafe content - show warning and hide content
            st.error(f"❌ **Unsafe**\n\nConfidence: {st.session_state.confidence_score:.4f}")
            st.warning("⚠️ Content has been flagged as unsafe and is hidden by default.")
            
            # Button to reveal blocked content
            if st.button("📋 Reveal Blocked Content", key="reveal_button"):
                show_blocked_content()
                st.rerun()
            
            # Show blocked content if button was clicked
            if st.session_state.show_blocked:
                st.subheader("Blocked Content (Hidden)")
                st.markdown("**Warning: The following content was flagged as unsafe:**")
                st.write(st.session_state.generated_content)
                st.info("This content was hidden because it violates safety guidelines.")
        
        # Optional: Add a divider after results
        st.markdown("---")
        st.caption("Generated by OPT-350M | Safety checked by DistilBERT")


# ============================================
# Main Function
# ============================================
def main():
    """
    Main function: Program entry point
    """
    render_ui()


# ============================================
# Program Entry Point
# ============================================
if __name__ == "__main__":
    main()
